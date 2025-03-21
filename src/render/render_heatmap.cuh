

extern "C" __global__
void kernel_render_heatmap(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, StageData* stagedatas
){
	auto grid     = cg::this_grid();
	auto block    = cg::this_thread_block();

	constexpr int PREFETCH_COUNT = 128;

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
	// iterations = min(iterations, 10);

	// uint32_t C = target.framebuffer[pixelID] & 0xffffffff;
	vec4 pixel = vec4{0.0f, 0.0f, 0.0f, 0.0f};
	float depth = Infinity;
	if(pixel_x < width && pixel_y < height){
		depth = __int_as_float(target.framebuffer[pixelID] >> 32);
	}
	float remainingOpacity = 1.0f;
	int numProcessed = 0;

	// iterate through all the splats
	// - 256 threads -> 256 splats at a time
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
			data.dir1 = normalize(basisvector1) / length(basisvector1);
			data.dir2 = normalize(basisvector2) / length(basisvector2);
			data.imgpos = vec2(stageData.imgPos_encoded) / 10.0f;
			data.depth = depth;
			// data.flags = stageData.flags;
			data.colors = vec4{
				((C >>  0) & 0xff) / 255.0f,
				((C >>  8) & 0xff) / 255.0f,
				((C >> 16) & 0xff) / 255.0f,
				((C >> 24) & 0xff) / 255.0f,
			};

			sh_rasterdata[threadIdx.x] = data;
		}

		int splatsInSharedMemory = min(numPointsInTile - PREFETCH_COUNT * iteration, PREFETCH_COUNT);

		__syncthreads();

		// now iterate with all threads of block through all splats.
		// i.e., all 256 threads process first splat, then all 256 threads process second, ...
		uint64_t t_start_draw = nanotime();
		for(int i = 0; i < splatsInSharedMemory; i++)
		{
			auto rasterdata = sh_rasterdata[i];
			vec2 imgCoords = rasterdata.imgpos;

			vec2 pFrag = {fpixel_x - imgCoords.x, fpixel_y - imgCoords.y};

			float sT = dot(rasterdata.dir1, pFrag); 
			float sB = dot(rasterdata.dir2, pFrag); 
			float w = (sT * sT + sB * sB);

			numProcessed++;

			// if(rasterdata.depth > depth) break;
			if(pixel_x >= width && pixel_y >= height) break;
			
			// Splat boundary at w = 1.0. The remaining part of the gaussian is discarded.
			if(w < 1.0f)
			{
				vec4 splatColor = rasterdata.colors;

				w = 1.0f - sqrt(w);
				remainingOpacity = 1.0f - pixel.w;

				if(args.uniforms.showRing && w < 0.05f){
					splatColor.a = 1.0f;
					w = 1.0f;
				}

				pixel.r += w * remainingOpacity * splatColor.a * splatColor.r * 255.0f;
				pixel.g += w * remainingOpacity * splatColor.a * splatColor.g * 255.0f;
				pixel.b += w * remainingOpacity * splatColor.a * splatColor.b * 255.0f;
				pixel.w += w * remainingOpacity * splatColor.a;
				
				// update depth buffer
				if(remainingOpacity < 0.030f)
				if((rasterdata.flags & FLAGS_DISABLE_DEPTHWRITE) == 0)
				if(pixel_x < width && pixel_y < height)
				{
					depth = rasterdata.depth;
					uint64_t udepth = __float_as_uint(depth);
					uint64_t pixel = (udepth << 32) | 0x00000000;
					
					target.framebuffer[pixelID] = pixel;
					
					// break;
				}
			}
		}

		__syncthreads();
	}

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


	// Transfer colors from register "pixel" to global memory. 
	if(pixel_x < width && pixel_y < height){
		uint32_t color = 0;
		uint8_t* rgba = (uint8_t*)&color;

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

		rgba[0] = numProcessed;
		rgba[3] = 255;

		// int cindex = numProcessed / 1000.0f;
		int cindex = log10(float(numProcessed)) / 2.0f;
		cindex = pow(float(numProcessed), 0.5f) * 0.05f;
		// cindex = pow(float(30'000), 0.5f) * 0.05f;
		cindex = clamp(cindex, 0, 10);

		rgba[0] = SPECTRAL[10 - cindex].x;
		rgba[1] = SPECTRAL[10 - cindex].y;
		rgba[2] = SPECTRAL[10 - cindex].z;

		if(args.uniforms.showTiles)
		if(tile_pixel_x == 15 || tile_pixel_y == 15)
		{
			rgba[0] = 0;
			rgba[1] = 0;
			rgba[2] = 0;
		}

		uint64_t pixelValue = target.framebuffer[pixelID];
		pixelValue = pixelValue & 0xffffffff'00000000;
		pixelValue = pixelValue | color;


		target.framebuffer[pixelID] = pixelValue;
		
	}

	// double nanos = double(nanotime() - t_start);
	// float millies = nanos / 1'000'000.0f;

	// if(millies > 7.2f && block.thread_rank() == 0){
	// 	printf("tile %d, splats %d, duration %.1f ms \n", tileID, numPointsInTile, millies);
	// }
}