
__device__ int g_numProcessedSubsets;

extern "C" __global__
void kernel_render_gaussians_subsets(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, StageData* stagedatas,
	uint64_t* subsetFramebuffers, 
	TileSubset* subsetsRoots, 
	TileSubset* subsetsList, 
	uint32_t* numSubsets,
	volatile uint8_t* opacities,
	uint32_t* opacityContributions

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

	// return;

	struct RasterizationData{
		vec2 dir1;
		vec2 dir2;
		vec2 imgpos;
		float depth;
		// uint32_t flags;
		vec4 colors;
	};

	__shared__ RasterizationData sh_rasterdata[PREFETCH_COUNT];

	uint32_t subsetID = grid.block_rank();

	if(subsetID >= *numSubsets) return;
	
	// Tile tile = tiles[tileID];
	TileSubset tile = subsetsList[subsetID];
	int tileID = tile.tile_x + tile.tile_y * tiles_x;

	// if(grid.thread_rank() == 0){
	// 	printf("%d \n", *numSubsets);
	// }

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
		// depth = __int_as_float(target.framebuffer[pixelID] >> 32);
	}
	float remainingTranslucency = 1.0f;

	// uint32_t remainingTranslucency_global = 255.0f - opacities[pixelID];

	// __shared__ uint32_t minRemaining;
	// uint32_t reduced = warpReduceMin(remainingTranslucency_global);
	// if(block.thread_rank() == 0){
	// 	minRemaining = reduced;
	// }
	// block.sync();
	// if(minRemaining < 10) return;


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

			// vec2 basisvector1 = decode_basisvector_i16vec2(stageData.basisvector1_encoded);
			// vec2 basisvector2 = decode_basisvector_i16vec2(stageData.basisvector2_encoded);
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

			if(rasterdata.depth > depth) break;
			if(pixel_x >= width && pixel_y >= height) break;
			
			// Splat boundary at w = 1.0. The remaining part of the gaussian is discarded.
			if(w < 1.0f)
			{
				vec4 splatColor = rasterdata.colors;

				// w = 1.0f - sqrt(w);
				w = exp(-0.5f * w * (8.0f));
				remainingTranslucency = 1.0f - pixel.w;

				if(args.uniforms.showRing && w < 0.05f){
					splatColor.a = 1.0f;
					w = 1.0f;
				}

				float alpha = w * remainingTranslucency * splatColor.a;

				pixel.r += alpha * splatColor.r * 255.0f;
				pixel.g += alpha * splatColor.g * 255.0f;
				pixel.b += alpha * splatColor.b * 255.0f;
				pixel.w += alpha;
				
				// // update depth buffer
				// if(remainingOpacity < 0.030f)
				// if((rasterdata.flags & FLAGS_DISABLE_DEPTHWRITE) == 0)
				// if(pixel_x < width && pixel_y < height)
				// {
				// 	depth = rasterdata.depth;
				// 	uint64_t udepth = __float_as_uint(depth);
				// 	uint64_t pixel = (udepth << 32) | 0x00000000;
					
				// 	target.framebuffer[pixelID] = pixel;
					
				// 	break;
				// }
			}
		}

		__syncthreads();
	}

	// __syncthreads();

	// bool isDebugTile = tile_x == 60 && tile_y == 50;
	// { 
	// 	uint8_t maxRemainingOpacity = clamp(blockReduceMax(int(256.0f * remainingOpacity)), 0, 256);



	// 	if(isDebugTile && block.thread_rank() == 0){
	// 		printf("maxRemainingOpacity: %d \n", maxRemainingOpacity);
	// 	}

	// }

	// if(isDebugTile && block.thread_rank() == 0){
	// 	printf("tile.subsetIndex: %d \n", tile.subsetIndex);
	// }

	{
		uint8_t remainingOpacity = clamp(255.0f * (1.0f - remainingTranslucency), 0.0f, 255.0f);

		int attempts = 0;

		do{

			attempts++;

			__shared__ bool gotLock;
			gotLock = false;

			block.sync();

			if(block.thread_rank() == 0){
				uint32_t old = atomicCAS(&opacityContributions[tileID], tile.subsetIndex, 0xffffffff);

				if(old == tile.subsetIndex){
					// lock obtained!
					gotLock = true;
				}
			}

			block.sync();

			if(gotLock){
				
				if(pixel_x < target.width && pixel_y < target.height){
					opacities[pixelID] = max(opacities[pixelID], remainingOpacity);
				}

				// unlock
				if(block.thread_rank() == 0){
					atomicExch(&opacityContributions[tileID], tile.subsetIndex + 1);
				}
				break;
			}else{
				__nanosleep(10);
				continue;
			}


		}while(attempts < 1000);
		
	}


	// Transfer colors from register "pixel" to global memory. 
	if(pixel_x < width && pixel_y < height){
		uint32_t color = 0;
		uint8_t* rgba = (uint8_t*)&color;

		if(args.uniforms.rendermode == RENDERMODE_COLOR){

			// color = target.framebuffer[pixelID] & 0xffffffff;

			if(args.uniforms.showTiles)
			if(tile_pixel_x == 15 || tile_pixel_y == 15)
			{
				pixel = pixel * 0.5f;
				pixel.w = 255.0f;
			}

			// uint32_t C = target.framebuffer[pixelID] & 0xffffffff;
			// vec4 oldPixel = vec4{
			// 	(C >>  0) & 0xff,
			// 	(C >>  8) & 0xff,
			// 	(C >> 16) & 0xff,
			// 	(C >> 24) & 0xff,
			// };

			rgba[0] = clamp(pixel.r, 0.0f, 255.0f) ;
			rgba[1] = clamp(pixel.g, 0.0f, 255.0f) ;
			rgba[2] = clamp(pixel.b, 0.0f, 255.0f) ;
			rgba[3] = clamp(255.0f * pixel.w, 0.0f, 255.0f) ;

			// if(isDebugTile) color = 0xff0000ff;

			// uint64_t pixelValue = target.framebuffer[pixelID];
			uint64_t pixelValue = color;


			// target.framebuffer[pixelID] = pixelValue;
			uint32_t localPixelID = tile_pixel_x + tile_pixel_y * 16;
			tile.subsetsFramebuffer[localPixelID] = pixelValue;

			
		}
	}

	// double nanos = double(nanotime() - t_start);
	// float millies = nanos / 1'000'000.0f;

	// if(millies > 7.2f && block.thread_rank() == 0){
	// 	printf("tile %d, splats %d, duration %.1f ms \n", tileID, numPointsInTile, millies);
	// }
}
