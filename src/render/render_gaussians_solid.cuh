
extern "C" __global__
void kernel_render_gaussians_solid(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, StageData* stagedatas,
	uint32_t pointsInTileThreshold
){
	auto grid     = cg::this_grid();
	auto block    = cg::this_thread_block();

	constexpr int PREFETCH_COUNT = 256;

	uint64_t t_start = nanotime();

	int width     = target.width;
	int height    = target.height;
	int numPixels = width * height;
	int tiles_x   = int(width + TILE_SIZE_3DGS - 1) / int(TILE_SIZE_3DGS);
	int tiles_y   = int(height + TILE_SIZE_3DGS - 1) / int(TILE_SIZE_3DGS);
	int numTiles  = tiles_x * tiles_y;

	struct RasterizationData{
		vec2 dir1;
		vec2 dir2;
		vec2 imgpos;
		float depth;
		// uint32_t flags;
		vec4 colors;
	};

	// if(grid.thread_rank() == 0){
	// 	printf("%u \n", sizeof(StageData));
	// }

	__shared__ RasterizationData sh_rasterdata[PREFETCH_COUNT];

	int tileID = blockIdx.x;
	
	Tile tile = tiles[tileID];

	// tile coordinates
	int tile_x = tileID % tiles_x;
	int tile_y = tileID / tiles_x;

	// this thread's pixel coordinate within tile
	int tilePixelIndex = threadIdx.x;
	int tile_pixel_x = tilePixelIndex % int(TILE_SIZE_3DGS);
	int tile_pixel_y = tilePixelIndex / int(TILE_SIZE_3DGS);

	// This thread's pixel coordinates within framebuffer
	int pixel_x = tile_x * TILE_SIZE_3DGS + tile_pixel_x;
	int pixel_y = tile_y * TILE_SIZE_3DGS + tile_pixel_y;
	float fpixel_x = pixel_x;
	float fpixel_y = pixel_y;
	int pixelID = int(pixel_x + pixel_y * width);

	int numPointsInTile = clamp(tile.lastIndex - tile.firstIndex + 1, 0u, 1'000'000u);

	if(block.thread_rank() == 0){
		atomicMax(&args.state->numSplatsInLargestTile, numPointsInTile);
	}

	int iterations = ((numPointsInTile + PREFETCH_COUNT - 1) / PREFETCH_COUNT);

	vec4 pixel = vec4{0.0f, 0.0f, 0.0f, 0.0f};
	float depth = Infinity;
	if(pixel_x < width && pixel_y < height){
		depth = __int_as_float(target.framebuffer[pixelID] >> 32);
	}
	float remainingTranslucency = 1.0f;

	// iterate through all the splats
	// - 16x16 pixels, 256 threads (1 thread per pixel)
	// - <PREFETCH_COUNT> splats at a time
	for(int iteration = 0; iteration < iterations; iteration++){

		int index = PREFETCH_COUNT * iteration + threadIdx.x;

		__syncthreads();

		// - load splats into shared memory
		// - each thread of the block loads one splat
		// - precompute stuff that is later used by all pixels/threads in tile
		if(index < numPointsInTile && block.thread_rank() < PREFETCH_COUNT){

			int splatIndex = indices[tile.firstIndex + index];

			auto stageData = stagedatas[splatIndex];
			
			uint32_t C = stageData.color;
			uint8_t* rgba = (uint8_t*)&C;

			// vec2 basisvector1 = decode_basisvector_i16vec2(stageData.basisvector1_encoded);
			// // vec2 basisvector2 = decode_basisvector_i16vec2(stageData.basisvector2_encoded);
			// float basisvec2_length = float(stageData.basisvector2_encoded) / 100.0f;
			// vec2 basisvector2 = normalize(vec2{-basisvector1.y, basisvector1.x}) * basisvec2_length;

			vec2 basisvector1, basisvector2;
			float depth;
			decode_stagedata(stageData, basisvector1, basisvector2, depth);

			RasterizationData data;
			data.dir1 = basisvector1 * (1.0f / dot(basisvector1, basisvector1));
			data.dir2 = basisvector2 * (1.0f / dot(basisvector2, basisvector2));
			data.imgpos = vec2(stageData.imgPos_encoded) / 10.0f;
			data.depth = depth;
			// data.flags = stageData.flags;
			data.colors = vec4{
				(C >>  0) & 0xff,
				(C >>  8) & 0xff,
				(C >> 16) & 0xff,
				((C >> 24) & 0xff) / 255.0f,
			};

			sh_rasterdata[threadIdx.x] = data;
		}

		int splatsInSharedMemory = min(numPointsInTile - PREFETCH_COUNT * iteration, PREFETCH_COUNT);

		block.sync();

		// now iterate with all threads of block through all splats.
		// i.e., all 256 threads process first splat, then all 256 threads process second, ...
		// uint64_t t_start_draw = nanotime();
		for(int i = 0; i < splatsInSharedMemory; i++)
		{
			auto rasterdata = sh_rasterdata[i];
			vec2 imgCoords = rasterdata.imgpos;

			vec2 pFrag = {fpixel_x - imgCoords.x, fpixel_y - imgCoords.y};

			float sT = dot(rasterdata.dir1, pFrag); 
			float sB = dot(rasterdata.dir2, pFrag); 
			float w = (sT * sT + sB * sB);

			if(rasterdata.depth > depth) break;
			if(pixel_x >= width || pixel_y >= height) break;
			
			// Splat boundary at w = 1.0. The remaining part of the gaussian is discarded.
			// if(w < 1.0f)

			if(w < 1.0f)
			{
				vec4 splatColor = rasterdata.colors;

				w = exp(-4.0f * w);
				
				// remainingTranslucency = 1.0f - pixel.w;

				splatColor.a = 1.0f;
				w = 1.0f;

				float alpha = w * (1.0f - pixel.w) * splatColor.a;

				pixel.r += alpha * splatColor.r;
				pixel.g += alpha * splatColor.g;
				pixel.b += alpha * splatColor.b;
				pixel.w += alpha;

				remainingTranslucency = 1.0f - pixel.w;
				
				// update depth buffer
				if(remainingTranslucency < 0.030f)
				// if(remainingTranslucency < 0.030f && (rasterdata.flags & FLAGS_DISABLE_DEPTHWRITE) == 0)
				{
					depth = rasterdata.depth;
					uint64_t udepth = __float_as_uint(depth);
					uint64_t pixel = (udepth << 32) | 0x00000000;
					
					target.framebuffer[pixelID] = pixel;
					
					break;
				}
			}
		}

		block.sync();
	}

	block.sync();

	// Transfer colors from register "pixel" to global memory. 
	if(pixel_x < width && pixel_y < height){
		uint32_t color = 0;
		uint8_t* rgba = (uint8_t*)&color;

		if(args.uniforms.rendermode == RENDERMODE_COLOR){

			color = target.framebuffer[pixelID] & 0xffffffff;

			if(args.uniforms.showTiles)
			if(tile_pixel_x == 15 || tile_pixel_y == 15)
			{
				pixel = pixel * 0.5f;
				pixel.w = 255.0f;
			}

			uint32_t C = target.framebuffer[pixelID] & 0xffffffff;
			vec4 oldPixel = vec4{
				(C >>  0) & 0xff,
				(C >>  8) & 0xff,
				(C >> 16) & 0xff,
				(C >> 24) & 0xff,
			};

			rgba[0] = clamp(pixel.r + remainingTranslucency * oldPixel.r, 0.0f, 255.0f) ;
			rgba[1] = clamp(pixel.g + remainingTranslucency * oldPixel.g, 0.0f, 255.0f) ;
			rgba[2] = clamp(pixel.b + remainingTranslucency * oldPixel.b, 0.0f, 255.0f) ;
			rgba[3] = 255;

			uint64_t pixelValue = target.framebuffer[pixelID];
			pixelValue = pixelValue & 0xffffffff'00000000;
			pixelValue = pixelValue | color;


			target.framebuffer[pixelID] = pixelValue;
		}else if(args.uniforms.rendermode == RENDERMODE_DEPTH){

			vec3 SPECTRAL[11] = {
				vec3{158,1,66},
				vec3{213,62,79},
				vec3{244,109,67},
				vec3{253,174,97},
				vec3{254,224,139},
				vec3{255,255,191},
				vec3{230,245,152},
				vec3{171,221,164},
				vec3{102,194,165},
				vec3{50,136,189},
				vec3{94,79,162},
			};

			float w = depth / 3.0f;
			float u = w - floor(w);

			int i0 = w;

			vec3 C0 = SPECTRAL[clamp((i0 + 0), 0, 10) % 11];
			vec3 C1 = SPECTRAL[clamp((i0 + 1), 0, 10) % 11];

			vec3 C = (1.0f - u) * C0 + u * C1;

			rgba[0] = C.x;
			rgba[1] = C.y;
			rgba[2] = C.z;

			uint32_t udepth = __float_as_uint(depth);

			uint64_t pixel = (uint64_t(udepth) << 32) | color;

			target.framebuffer[pixelID] = pixel;
		}
	}
}




// extern "C" __global__
// void kernel_render_gaussians_half(
// 	CommonLaunchArgs args, RenderTarget target,
// 	Tile* tiles, uint32_t* indices, StageData* stagedatas,
// 	bool doProfiling, DbgProfileTime* times, uint32_t* timesCount
// 	// DbgProfileUtilization* utilization
// ){
// 	auto grid     = cg::this_grid();
// 	auto block    = cg::this_thread_block();

// 	constexpr int PREFETCH_COUNT = 128;

// 	// uint64_t t_start = nanotime();

// 	int width     = target.width;
// 	int height    = target.height;
// 	int numPixels = width * height;
// 	int tiles_x   = int(width + TILE_SIZE_3DGS - 1) / int(TILE_SIZE_3DGS);
// 	int tiles_y   = int(height + TILE_SIZE_3DGS - 1) / int(TILE_SIZE_3DGS);
// 	int numTiles  = tiles_x * tiles_y;

// 	struct RasterizationData{
// 		__half2 dir1;
// 		__half2 dir2;
// 		vec2 imgpos;
// 		float depth;
// 		uint32_t flags;
// 		__half colors[4];
// 	};

// 	__shared__ RasterizationData sh_rasterdata[PREFETCH_COUNT];

// 	int tileID = blockIdx.x;
	
// 	Tile tile = tiles[tileID];

// 	// tile coordinates
// 	int tile_x = tileID % tiles_x;
// 	int tile_y = tileID / tiles_x;

// 	// this thread's pixel coordinate within tile
// 	int tilePixelIndex = threadIdx.x;
// 	int tile_pixel_x = tilePixelIndex % int(TILE_SIZE_3DGS);
// 	int tile_pixel_y = tilePixelIndex / int(TILE_SIZE_3DGS);

// 	// This thread's pixel coordinates within framebuffer
// 	int pixel_x = tile_x * TILE_SIZE_3DGS + tile_pixel_x;
// 	int pixel_y = tile_y * TILE_SIZE_3DGS + tile_pixel_y;
// 	float fpixel_x = pixel_x;
// 	float fpixel_y = pixel_y;
// 	int pixelID = int(pixel_x + pixel_y * width);

// 	int numPointsInTile = tile.lastIndex - tile.firstIndex + 1;
// 	if(tile.firstIndex > tile.lastIndex) {
// 		numPointsInTile = 0;
// 	}

// 	if(block.thread_rank() == 0){
// 		atomicMax(&args.state->numSplatsInLargestTile, numPointsInTile);
// 	}

// 	int iterations = ((numPointsInTile + PREFETCH_COUNT - 1) / PREFETCH_COUNT);

// 	__half pixel[4] = {0.0f, 0.0f, 0.0f, 0.0f};
// 	float depth = Infinity;
// 	if(pixel_x < width && pixel_y < height){
// 		depth = __int_as_float(target.framebuffer[pixelID] >> 32);
// 	}
// 	__half remainingTranslucency = 1.0f;

// 	// iterate through all the splats
// 	// - 16x16 pixels, 256 threads (1 thread per pixel)
// 	// - <PREFETCH_COUNT> splats at a time
// 	for(int iteration = 0; iteration < iterations; iteration++){

// 		int index = PREFETCH_COUNT * iteration + threadIdx.x;

// 		// if(iteration > 50){
// 		// 	if(iteration % 2 == 0) iteration++;
// 		// }

// 		__syncthreads();

// 		// - load splats into shared memory
// 		// - each thread of the block loads one splat
// 		// - precompute stuff that is later used by all pixels/threads in tile
// 		if(index < numPointsInTile && block.thread_rank() < PREFETCH_COUNT){

// 			int splatIndex = indices[tile.firstIndex + index];

// 			auto stageData = stagedatas[splatIndex];
			
// 			uint32_t C = stageData.color;
// 			uint8_t* rgba = (uint8_t*)&C;

// 			__half2 basisvector1 = decode_basisvector_i16vec2_half2(stageData.basisvector1_encoded);
// 			__half2 basisvector2 = decode_basisvector_i16vec2_half2(stageData.basisvector2_encoded);

// 			RasterizationData data;
		
// 			__half dota = basisvector1.x * basisvector1.x + basisvector1.y * basisvector1.y;
// 			__half dotb = basisvector2.x * basisvector2.x + basisvector2.y * basisvector2.y;
// 			data.dir1 = {
// 				basisvector1.x / dota,
// 				basisvector1.y / dota,
// 			};
// 			data.dir2 = {
// 				basisvector2.x / dotb,
// 				basisvector2.y / dotb,
// 			};

// 			data.imgpos = vec2(stageData.imgPos_encoded) / 10.0f;
// 			data.depth = stageData.depth;
// 			data.flags = stageData.flags;
// 			data.colors[0] = (C >>   0) & 0xff;
// 			data.colors[1] = (C >>   8) & 0xff;
// 			data.colors[2] = (C >>  16) & 0xff;
// 			data.colors[3] = ((C >> 24) & 0xff) / 255.0f;

// 			sh_rasterdata[threadIdx.x] = data;
// 		}

// 		int splatsInSharedMemory = min(numPointsInTile - PREFETCH_COUNT * iteration, PREFETCH_COUNT);

// 		block.sync();

// 		// now iterate with all threads of block through all splats.
// 		// i.e., all 256 threads process first splat, then all 256 threads process second, ...
// 		// uint64_t t_start_draw = nanotime();
// 		for(int i = 0; i < splatsInSharedMemory; i++)
// 		{
// 			auto rasterdata = sh_rasterdata[i];
// 			vec2 imgCoords = rasterdata.imgpos;

// 			__half2 pFrag = {fpixel_x - imgCoords.x, fpixel_y - imgCoords.y};
// 			__half sT = rasterdata.dir1.x * pFrag.x + rasterdata.dir1.y * pFrag.y;
// 			__half sB = rasterdata.dir2.x * pFrag.x + rasterdata.dir2.y * pFrag.y;
// 			__half w = (sT * sT + sB * sB);

// 			if(rasterdata.depth > depth) break;
// 			if(pixel_x >= width || pixel_y >= height) break;
			
// 			// Splat boundary at w = 1.0. The remaining part of the gaussian is discarded.
// 			if(w < static_cast<__half>(1.0f))
// 			{
// 				__half* splatColor = rasterdata.colors;

// 				w = hexp(w * static_cast<__half>(-4.0));
				
// 				remainingTranslucency = static_cast<__half>(1.0) - pixel[3];

// 				if(args.uniforms.showRing && w < static_cast<__half>(0.05f)){
// 					splatColor[3] = 1.0f;
// 					w = 1.0f;
// 				}

// 				__half alpha = w * remainingTranslucency * splatColor[3];

// 				pixel[0] += alpha * splatColor[0];
// 				pixel[1] += alpha * splatColor[1];
// 				pixel[2] += alpha * splatColor[2];
// 				pixel[3] += alpha;
				
// 				// update depth buffer
// 				if(remainingTranslucency < static_cast<__half>(0.030) && (rasterdata.flags & FLAGS_DISABLE_DEPTHWRITE) == 0)
// 				{
// 					depth = rasterdata.depth;
// 					uint64_t udepth = __float_as_uint(depth);
// 					uint64_t pixel = (udepth << 32) | 0x00000000;
					
// 					target.framebuffer[pixelID] = pixel;
					
// 					break;
// 				}
// 			}
// 		}

// 		block.sync();
// 	}

// 	block.sync();

// 	// Transfer colors from register "pixel" to global memory. 
// 	if(pixel_x < width && pixel_y < height){
// 		uint32_t color = 0;
// 		uint8_t* rgba = (uint8_t*)&color;

// 		if(args.uniforms.rendermode == RENDERMODE_COLOR || args.uniforms.rendermode == RENDERMODE_SPLATLETS){

// 			color = target.framebuffer[pixelID] & 0xffffffff;

// 			if(args.uniforms.showTiles)
// 			if(tile_pixel_x == 15 || tile_pixel_y == 15)
// 			{
// 				pixel[0] = pixel[0] * static_cast<__half>(0.5f);
// 				pixel[1] = pixel[1] * static_cast<__half>(0.5f);
// 				pixel[2] = pixel[2] * static_cast<__half>(0.5f);
// 				pixel[3] = static_cast<__half>(255.0f);
// 			}

// 			uint32_t C = target.framebuffer[pixelID] & 0xffffffff;
// 			vec4 oldPixel = vec4{
// 				(C >>  0) & 0xff,
// 				(C >>  8) & 0xff,
// 				(C >> 16) & 0xff,
// 				(C >> 24) & 0xff,
// 			};

// 			rgba[0] = clamp(float(pixel[0]) + float(remainingTranslucency) * oldPixel.r, 0.0f, 255.0f);
// 			rgba[1] = clamp(float(pixel[1]) + float(remainingTranslucency) * oldPixel.g, 0.0f, 255.0f);
// 			rgba[2] = clamp(float(pixel[2]) + float(remainingTranslucency) * oldPixel.b, 0.0f, 255.0f);
// 			rgba[3] = 255;

// 			uint64_t pixelValue = target.framebuffer[pixelID];
// 			pixelValue = pixelValue & 0xffffffff'00000000;
// 			pixelValue = pixelValue | color;


// 			target.framebuffer[pixelID] = pixelValue;
// 		}else if(args.uniforms.rendermode == RENDERMODE_DEPTH){

// 			vec3 SPECTRAL[11] = {
// 				vec3{158,1,66},
// 				vec3{213,62,79},
// 				vec3{244,109,67},
// 				vec3{253,174,97},
// 				vec3{254,224,139},
// 				vec3{255,255,191},
// 				vec3{230,245,152},
// 				vec3{171,221,164},
// 				vec3{102,194,165},
// 				vec3{50,136,189},
// 				vec3{94,79,162},
// 			};

// 			float w = depth / 3.0f;
// 			float u = w - floor(w);

// 			int i0 = w;

// 			vec3 C0 = SPECTRAL[clamp((i0 + 0), 0, 10) % 11];
// 			vec3 C1 = SPECTRAL[clamp((i0 + 1), 0, 10) % 11];

// 			vec3 C = (1.0f - u) * C0 + u * C1;

// 			rgba[0] = C.x;
// 			rgba[1] = C.y;
// 			rgba[2] = C.z;

// 			uint32_t udepth = __float_as_uint(depth);

// 			uint64_t pixel = (uint64_t(udepth) << 32) | color;

// 			target.framebuffer[pixelID] = pixel;
// 		}
// 	}

// 	// if(block.thread_rank() == 0 && doProfiling)
// 	// {
// 	// 	DbgProfileTime time;
// 	// 	time.t_start = t_start;
// 	// 	time.t_end = nanotime();

// 	// 	uint32_t index = atomicAdd(timesCount, 1);

// 	// 	times[index] = time;
// 	// }

// 	// double nanos = double(nanotime() - t_start);
// 	// float millies = nanos / 1'000'000.0f;

// 	// if(args.uniforms.frameCount % 100 == 0)
// 	// // if(millies > 4.0f && block.thread_rank() == 0)
// 	// if(isDebugTile(tile_x, tile_y) && block.thread_rank() == 0)
// 	// {
// 	// 	// printf("tile %d, splats %d, duration %.1f ms \n", tileID, numPointsInTile, millies);
// 	// 	// printf("millies: %.1f ms, splats: %d k \n", millies, numPointsInTile / 1000);

// 	// 	int splatsMs = float(numPointsInTile) / millies;
// 	// 	printf("millies: %.1f ms, splats: %3d k, %d k splats/ms \n", millies, numPointsInTile / 1000, splatsMs / 1000);
// 	// }
// }
