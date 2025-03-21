
__device__ float dbg_contributions[100'000];

extern "C" __global__
void kernel_render_gaussians_fetchfilter(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, StageData* stagedatas
){
	auto grid     = cg::this_grid();
	auto block    = cg::this_thread_block();

	constexpr int PREFETCH_COUNT = 128;

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
		uint32_t flags;
		vec4 colors;
	};

	__shared__ RasterizationData sh_rasterdata[PREFETCH_COUNT];
	__shared__ float sh_maxRemainingOpacity;

	sh_maxRemainingOpacity = 1.0;

	int tileID = blockIdx.x;
	
	Tile tile = tiles[tileID];

	// tile coordinates
	int tile_x = tileID % tiles_x;
	int tile_y = tileID / tiles_x;
	// int tileID = tile_x + tile_y * tiles_x;

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

	int numPointsInTile = tile.lastIndex - tile.firstIndex + 1;
	if(tile.firstIndex > tile.lastIndex) {
		numPointsInTile = 0;
		// return;
	}

	int iterations = ((numPointsInTile + PREFETCH_COUNT - 1) / PREFETCH_COUNT);
	// iterations = min(iterations, 20);

	// uint32_t C = target.framebuffer[pixelID] & 0xffffffff;
	vec4 pixel = vec4{0.0f, 0.0f, 0.0f, 0.0f};
	float depth = Infinity;
	if(pixel_x < width && pixel_y < height){
		depth = __int_as_float(target.framebuffer[pixelID] >> 32);
	}
	float remainingOpacity = 1.0f;

	if(isDebugTile(tile_x, tile_y))
	{ // DEBUG - reset contributions

		for(
			int i = block.thread_rank();
			i < numPointsInTile; 
			i += block.num_threads()
		){
			dbg_contributions[i] = 0.0f;
		}
	}

	// iterate through all the splats
	// - 16x16 pixels, 256 threads (1 thread per pixel)
	// - <PREFETCH_COUNT> splats at a time
	for(int iteration = 0; iteration < iterations; iteration++){

		int index = PREFETCH_COUNT * iteration + threadIdx.x;

		// if(iteration > 50){
		// 	if(iteration % 2 == 0) iteration++;
		// }

		__syncthreads();

		// - load splats into shared memory
		// - each thread of the block loads one splat
		// - precompute stuff that is later used by all pixels/threads in tile
		if(index < numPointsInTile && block.thread_rank() < PREFETCH_COUNT){

			int splatIndex = indices[tile.firstIndex + index];

			auto stageData = stagedatas[splatIndex];
			
			uint32_t C = stageData.color;
			uint8_t* rgba = (uint8_t*)&C;

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


			// // DEBUG
			// if(index == 4){
			// 	// imgCoords.x -= 8.0f;
			// 	// imgCoords.y -= 5.0f;


			// 	if(isDebugTile(tile_x, tile_y))
			// 	// if(block.thread_rank() == 0)
			// 	{
			// 		// printf("splatIndex: %d \n", splatIndex);
			// 		// printf("opacity: %f \n", 255.0f * rasterdata.colors.a);

			// 		// printf("imgCoords: %f, %f \n", data.imgpos.x, data.imgpos.y);
			// 		// printf("basisvector1: %f, %f \n", basisvector1.x, basisvector1.y);
			// 		// printf("basisvector2: %f, %f \n", basisvector2.x, basisvector2.y);
			// 	}

			// 	data.imgpos += vec2{0.0f, -8.0f};
			// }

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

			// if(remainingOpacity < 0.030f) break;
			if(rasterdata.depth > depth) break;
			if(pixel_x >= width || pixel_y >= height) break;
			
			// Splat boundary at w = 1.0. The remaining part of the gaussian is discarded.
			if(w < 1.0f)
			// if(PREFETCH_COUNT * iteration + i == 4)
			{
				vec4 splatColor = rasterdata.colors;

				w = 1.0f - sqrt(w);
				
				remainingOpacity = 1.0f - pixel.w;

				if(args.uniforms.showRing && w < 0.05f){
					splatColor.a = 1.0f;
					w = 1.0f;
				}

				// w = 1.0f;
				// splatColor.a = 1.0f;

				float alpha = w * remainingOpacity * splatColor.a;
				// float alpha = w * splatColor.a;
				pixel.r += alpha * splatColor.r;
				pixel.g += alpha * splatColor.g;
				pixel.b += alpha * splatColor.b;
				pixel.w += alpha;

				// DEBUG
				if(isDebugTile(tile_x, tile_y)){
					atomicAdd(&dbg_contributions[PREFETCH_COUNT * iteration + i], alpha);
				}
				
				// update depth buffer
				// if(remainingOpacity < 0.030f && (rasterdata.flags & FLAGS_DISABLE_DEPTHWRITE) == 0)
				// {
				// 	depth = rasterdata.depth;
				// 	uint64_t udepth = __float_as_uint(depth);
				// 	uint64_t pixel = (udepth << 32) | 0x00000000;
					
				// 	target.framebuffer[pixelID] = pixel;
					
				// 	break;
				// }
			}
		}

		block.sync();
	}

	block.sync();

	// DEBUG
	if(args.uniforms.frameCount % 200 == 0)
	if(isDebugTile(tile_x, tile_y)){

		if(block.thread_rank() == 0){
			
			printf("===================================\n");
			printf("contributions: \n");
			// for(int i = 0; i < min(numPointsInTile, 100); i++)
			int start = 0;
			int size = numPointsInTile;
			int numChecked = 0;
			int numZero = 0;

			for(int i = start; i < min(numPointsInTile, start + size); i++){
				if(i < 40){
					printf("%4d  ", i);
				}
			}
			printf("\n");

			for(int i = start; i < min(numPointsInTile, start + size); i++)
			{
				float contribution = dbg_contributions[i];

				if(i < 40){
					printf("%4d, ", int(255.0f * contribution));
				}

				// if(contribution > 0.1f){
				// 	printf(" <%4d, %.1f> ", i, contribution);
				// }

				if(contribution == 0.0f){
					numZero++;
				}
				numChecked++;
			}

			int i = start;
			
			while(i < size){
				
				int numContributors = 0;
				float acc = 0.0f;
				int mergesize = 1000;
				int target = i + mergesize;
				for(;i < target; i++){
					float contribution = dbg_contributions[i];
					acc += contribution;
					if(contribution > 0.0f) numContributors++;
				}

				// printf("%.1f, ", acc);
				// printf("%4d, ", numContributors);

			}

			printf("\n");
			printf("checked: %4d, zeros: %4d \n", numChecked, numZero);
		}

	}


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

			rgba[0] = clamp(pixel.r + remainingOpacity * oldPixel.r, 0.0f, 255.0f) ;
			rgba[1] = clamp(pixel.g + remainingOpacity * oldPixel.g, 0.0f, 255.0f) ;
			rgba[2] = clamp(pixel.b + remainingOpacity * oldPixel.b, 0.0f, 255.0f) ;
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

	// double nanos = double(nanotime() - t_start);
	// float millies = nanos / 1'000'000.0f;

	// if(args.uniforms.frameCount % 100 == 0)
	// // if(millies > 4.0f && block.thread_rank() == 0)
	// if(isDebugTile(tile_x, tile_y) && block.thread_rank() == 0)
	// {
	// 	// printf("tile %d, splats %d, duration %.1f ms \n", tileID, numPointsInTile, millies);
	// 	// printf("millies: %.1f ms, splats: %d k \n", millies, numPointsInTile / 1000);

	// 	int splatsMs = float(numPointsInTile) / millies;
	// 	printf("millies: %.1f ms, splats: %3d k, %d k splats/ms \n", millies, numPointsInTile / 1000, splatsMs / 1000);
	// }
}
