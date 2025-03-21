constexpr int PREFETCH_COUNT = 128;

#define DEBUGING
// #define DEBUG_MEASURE
// #define DEBUG_HEATMAP

// Padded to 64 bytes to avoid bank-conflicts. (16 banks of 4 byte words -> 64 bytes)
struct RasterizationData{
	vec2 dir1;
	vec2 dir2;
	vec2 imgpos;
	float depth;
	uint32_t flags;
	vec4 colors;
	uint32_t padding1;
	uint32_t padding2;
	uint32_t padding3;
	uint32_t padding4;
	// uint32_t padding5;
};

extern "C" __global__
void kernel_render_gaussians(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, StageData* stagedatas,
	uint32_t pointsInTileThreshold
){
	auto grid     = cg::this_grid();
	auto block    = cg::this_thread_block();

	// constexpr int PREFETCH_COUNT = 128;

	uint64_t t_start = nanotime();

	int width     = target.width;
	int height    = target.height;
	int numPixels = width * height;
	int tiles_x   = int(width + TILE_SIZE_3DGS - 1) / TILE_SIZE_3DGS;
	int tiles_y   = int(height + TILE_SIZE_3DGS - 1) / TILE_SIZE_3DGS;
	int numTiles  = tiles_x * tiles_y;

	__shared__ RasterizationData sh_rasterdata[PREFETCH_COUNT];
	__shared__ float sh_remainingTranslucency[256];

	int tileID = grid.block_rank();
	
	Tile tile = tiles[tileID];

	// tile coordinates
	int tile_x = tileID % tiles_x;
	int tile_y = tileID / tiles_x;

	// this thread's pixel coordinate within tile
	int tilePixelIndex = threadIdx.x;
	int tile_pixel_x = tilePixelIndex % TILE_SIZE_3DGS;
	int tile_pixel_y = tilePixelIndex / TILE_SIZE_3DGS;
	// int tile_pixel_x = 0;
	// int tile_pixel_y = 0;
	// if(tilePixelIndex < 64){
	// 	tile_pixel_x = 0 + (tilePixelIndex -   0) % 8;
	// 	tile_pixel_y = 0 + (tilePixelIndex -   0) / 8;
	// }else if(tilePixelIndex < 128){
	// 	tile_pixel_x = 0 + (tilePixelIndex -  64) % 8;
	// 	tile_pixel_y = 8 + (tilePixelIndex -  64) / 8;
	// }else if(tilePixelIndex < 192){
	// 	tile_pixel_x = 8 + (tilePixelIndex - 128) % 8;
	// 	tile_pixel_y = 0 + (tilePixelIndex - 128) / 8;
	// }else{
	// 	tile_pixel_x = 8 + (tilePixelIndex - 192) % 8;
	// 	tile_pixel_y = 8 + (tilePixelIndex - 192) / 8;
	// }

	// This thread's pixel coordinates within framebuffer
	int pixel_x = tile_x * TILE_SIZE_3DGS + tile_pixel_x;
	int pixel_y = tile_y * TILE_SIZE_3DGS + tile_pixel_y;
	float fpixel_x = pixel_x;
	float fpixel_y = pixel_y;
	int pixelID = int(pixel_x + pixel_y * width);

	int numPointsInTile = clamp(tile.lastIndex - tile.firstIndex + 1, 0u, 1'000'000u);

	if(numPointsInTile > pointsInTileThreshold) return;

	// __shared__ uint32_t dbg_numAccepted;
	// __shared__ uint32_t dbg_numRejected;

	// if(block.thread_rank() == 0){
	// 	atomicMax(&args.state->numSplatsInLargestTile, numPointsInTile);
	// 	dbg_numAccepted = 0;
	// 	dbg_numRejected = 0;
	// }

	int iterations = ((numPointsInTile + PREFETCH_COUNT - 1) / PREFETCH_COUNT);
	uint32_t numProcessed = 0;

	vec4 pixel = vec4{0.0f, 0.0f, 0.0f, 0.0f};
	float depth = Infinity;
	if(pixel_x < width && pixel_y < height){
		depth = __int_as_float(target.framebuffer[pixelID] >> 32);
	}
	float remainingTranslucency = 1.0f;
	sh_remainingTranslucency[block.thread_rank()] = 1.0f;

	// iterate through all the splats
	// - 16x16 pixels, 256 threads (1 thread per pixel)
	// - <PREFETCH_COUNT> splats at a time
	for(int iteration = 0; iteration < iterations; iteration++){


		// if(block.thread_rank() == 0){
		// 	sh_numRejected = 0;
		// 	memset(&sh_numRejected_warpwise, 0, 4 * 8);
		// }
		// if(block.thread_rank() % 32 == 0){
		// 	sh_numRejected_warpwise[block.thread_rank() / 32] = 0;
		// }

		// block.sync();

		// if(iteration > 100) continue;
		// if(iteration > 10){
		// 	if(iteration % 2 == 0) continue;
		// }

		//-------------------------------------------
		//   MULTIWARP-START
		//-------------------------------------------

		// - load splats into shared memory
		// - each thread of the block loads one splat
		// - precompute stuff that is later used by all pixels/threads in tile
		int index = PREFETCH_COUNT * iteration + block.thread_rank();
		if(index < numPointsInTile && block.thread_rank() < PREFETCH_COUNT){


			#if defined(FRAGWISE_ORDERING)
				StageData stageData = stagedatas[tile.firstIndex + index];
			#else
				int splatIndex = indices[tile.firstIndex + index];
				StageData stageData = stagedatas[splatIndex];
			#endif
			
			uint32_t C = stageData.color;

			vec2 basisvector1, basisvector2;
			float depth;
			decode_stagedata(stageData, basisvector1, basisvector2, depth);

			// TODO: not a normalized direction vector...rename?
			RasterizationData data;
			data.dir1 = basisvector1 / dot(basisvector1, basisvector1);
			data.dir2 = basisvector2 / dot(basisvector2, basisvector2);
			data.imgpos = vec2(stageData.imgPos_encoded) / 10.0f;
			data.depth = depth;
			data.flags = stageData.flags;
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
			vec4 splatColor = rasterdata.colors;
			vec2 imgCoords = rasterdata.imgpos;

			numProcessed++;

			// if(splatColor.a < 0.003f) continue;

			vec2 pFrag = {fpixel_x - imgCoords.x, fpixel_y - imgCoords.y};

			float sT = dot(rasterdata.dir1, pFrag); 
			float sB = dot(rasterdata.dir2, pFrag); 
			float w = (sT * sT + sB * sB);

			if(rasterdata.depth > depth) break;
			if(pixel_x >= width || pixel_y >= height) break;
			
			// Splat boundary at w = 1.0. The remaining part of the gaussian is discarded.
			if(w < 1.0f)
			{
				// vec4 splatColor = rasterdata.colors;

				w = exp(-4.0f * w);
				
				// remainingTranslucency = 1.0f - pixel.w;

				if(args.uniforms.showRing && w < 0.025f){
					splatColor.a = 1.0f;
					w += 0.5f;
				}

				float alpha = w * (1.0f - pixel.w) * splatColor.a;

				pixel.r += alpha * splatColor.r;
				pixel.g += alpha * splatColor.g;
				pixel.b += alpha * splatColor.b;
				pixel.w += alpha;

				remainingTranslucency = 1.0f - pixel.w;
				
				// update depth buffer
				// if(remainingTranslucency < 8.0f / 255.0f)
				// if(remainingTranslucency < 0.030f)
				if(remainingTranslucency < 8.0f / 255.0f && (rasterdata.flags & FLAGS_DISABLE_DEPTHWRITE) == 0)
				{
					depth = rasterdata.depth;
					uint64_t udepth = __float_as_uint(depth);
					uint64_t pixel = (udepth << 32) | 0x00000000;
					
					target.framebuffer[pixelID] = pixel;
					
					break;
				}
			}
		}

		sh_remainingTranslucency[block.thread_rank()] = remainingTranslucency;

		block.sync();
	}

	block.sync();

	// if(isDebugTile(tile_x, tile_y)){
	// 	// args.state->dbg.numAccepted = dbg_numAccepted;
	// 	// args.state->dbg.numRejected = dbg_numRejected;
	// 	if(block.thread_rank() == 0){
	// 		printf("numPointsInTile: %d\n", numPointsInTile);
	// 	}
	// }

	#if defined(DEBUG_MEASURE)
	float millies = float(nanotime() - t_start) / 1'000'000.0f;
	#endif

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

			// if(millies > 2.0f){
			// 	color = 0xff0000ff;
			// }

			#if defined(DEBUG_HEATMAP)
			float w = log2(pow(float(numProcessed), 0.7f));
			uint32_t spectralIndex = clamp(int(w), 0, 10);

			// if(numProcessed > 1000)
			{
				float3 cf = SPECTRAL[10 - spectralIndex];
				rgba[0] = cf.x;
				rgba[1] = cf.y;
				rgba[2] = cf.z;
			}
			#endif

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

	// uint64_t nanos = nanotime() - t_start;
	// if(nanos > 5'500'000){
	// 	float millies = float(nanos) / 1'000'000.0f;

	// 	if(block.thread_rank() == 0){
	// 		printf("millies: %.1f ms\n", millies);
	// 	}
	// }
}

bool areSimilar(RasterizationData splat1, RasterizationData splat2){

	// vec2 diff_pos = length(splat1.imgpos - splat2.imgpos);
	float angle1 = atan2(splat1.dir1.y, splat1.dir1.x);
	float angle2 = atan2(splat2.dir2.y, splat2.dir2.x);

	// let length_1a = splat1.basisvector1.length();
	// let length_1b = splat1.basisvector2.length();
	// let length_2a = splat2.basisvector1.length();
	// let length_2b = splat2.basisvector2.length();

	float diff_angle = abs(angle1 - angle2);

	float threshold_length  = 3.0f;
	float threshold_degrees = 20.0f;
	float threshold_radians = 3.1415f * threshold_degrees / 180.0f;
	float threshold_pos     = 2.0f;

	// let similarLengthA = Math.abs(length_1a - length_2a) < threshold_length;
	// let similarLengthB = Math.abs(length_1b - length_2b) < threshold_length;

	// if(splat2.opacity >= 0.5) return false;

	// if(!similarLengthA) return false;
	// if(!similarLengthB) return false;
	if(diff_angle > threshold_radians) return false;
	// if(diff_pos > threshold_pos) return false;

	return true;
}

extern "C" __global__
void kernel_render_gaussians_with_discard(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, StageData* stagedatas,
	uint32_t pointsInTileThreshold
){
	auto grid     = cg::this_grid();
	auto block    = cg::this_thread_block();

	// return;

	int width     = target.width;
	int height    = target.height;
	int numPixels = width * height;
	int tiles_x   = int(width + TILE_SIZE_3DGS - 1) / int(TILE_SIZE_3DGS);
	int tiles_y   = int(height + TILE_SIZE_3DGS - 1) / int(TILE_SIZE_3DGS);
	int numTiles  = tiles_x * tiles_y;

	__shared__ RasterizationData sh_rasterdata[PREFETCH_COUNT];
	__shared__ RasterizationData sh_rasterdata_coalesced[PREFETCH_COUNT];
	__shared__ float sh_remainingTranslucency[256];

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

	if(numPointsInTile <= pointsInTileThreshold) return;

	__shared__ uint32_t dbg_numAccepted;
	__shared__ uint32_t dbg_numRejected;

	if(block.thread_rank() == 0){
		atomicMax(&args.state->numSplatsInLargestTile, numPointsInTile);
		dbg_numAccepted = 0;
		dbg_numRejected = 0;
	}

	int iterations = ((numPointsInTile + PREFETCH_COUNT - 1) / PREFETCH_COUNT);

	vec4 pixel = vec4{0.0f, 0.0f, 0.0f, 0.0f};
	float depth = Infinity;
	if(pixel_x < width && pixel_y < height){
		depth = __int_as_float(target.framebuffer[pixelID] >> 32);
	}
	float remainingTranslucency = 1.0f;
	sh_remainingTranslucency[block.thread_rank()] = 1.0f;
	__shared__ uint32_t sh_numRejected;
	__shared__ uint32_t sh_numRejected_warpwise[8]; // 256 threads per block / 32 threads per warp = 8 warps
	__shared__ uint32_t sh_acceptanceMasks[(PREFETCH_COUNT + 32 - 1) / 32];
	__shared__ uint32_t sh_numAccepted[(PREFETCH_COUNT + 32 - 1) / 32];
	__shared__ uint32_t sh_prefixSum[(PREFETCH_COUNT + 32 - 1) / 32];

	#ifdef DEBUGING
	__shared__ uint32_t numRejected;

	if(block.thread_rank() == 0){
		numRejected = 0;
	}
	#endif

	// iterate through all the splats
	// - 16x16 pixels, 256 threads (1 thread per pixel)
	// - <PREFETCH_COUNT> splats at a time
	for(int iteration = 0; iteration < iterations; iteration++){

		// if(iteration > 200) break;

		if(block.thread_rank() == 0){
			sh_acceptanceMasks[0] = 0;
			sh_acceptanceMasks[1] = 0;
			sh_acceptanceMasks[2] = 0;
			sh_acceptanceMasks[3] = 0;
			sh_numAccepted[0] = 0;
			sh_numAccepted[1] = 0;
			sh_numAccepted[2] = 0;
			sh_numAccepted[3] = 0;
			sh_prefixSum[0] = 0;
			sh_prefixSum[1] = 0;
			sh_prefixSum[2] = 0;
			sh_prefixSum[3] = 0;
		}

		block.sync();

		// LOAD <PREFETCH_COUNT> SPLATS
		int index = PREFETCH_COUNT * iteration + block.thread_rank();
		if(index < numPointsInTile && block.thread_rank() < PREFETCH_COUNT)
		{
			auto warp = cg::coalesced_threads();
			
			int splatIndex = indices[tile.firstIndex + index];
			StageData stageData = stagedatas[splatIndex];

			uint32_t C = stageData.color;

			vec2 basisvector1, basisvector2;
			float depth;
			decode_stagedata(stageData, basisvector1, basisvector2, depth);

			RasterizationData data;
			data.dir1 = basisvector1 / dot(basisvector1, basisvector1);
			data.dir2 = basisvector2 / dot(basisvector2, basisvector2);
			data.imgpos = vec2(stageData.imgPos_encoded) / 10.0f;
			data.depth = depth;
			data.flags = stageData.flags;
			data.colors = vec4{
				(C >>  0) & 0xff,
				(C >>  8) & 0xff,
				(C >> 16) & 0xff,
				((C >> 24) & 0xff) / 255.0f,
			};

			sh_rasterdata[block.thread_rank()] = data;

			// check if splat should be discarded
			bool isRejected = false;
			bool hasLowOpacity = data.colors.a < 0.8f;
			bool isSmall = length(basisvector1) < 110.0f && length(basisvector2) < 110.0f;


			// if(warp.thread_rank() % 2 == 1)
			{
				float angle = atan2(data.dir1.y, data.dir1.x);
				float angle_prev = warp.shfl(angle, warp.thread_rank() - 1);
				float diff_angle = abs(angle - angle_prev);

				float px_prev = warp.shfl(data.imgpos.x, warp.thread_rank() - 1);
				float py_prev = warp.shfl(data.imgpos.y, warp.thread_rank() - 1);
				float diff_pos = length(vec2(px_prev, py_prev) - data.imgpos);

				float threshold_degrees = 30.0f;
				float threshold_radians = 3.1415f * threshold_degrees / 180.0f;
				float threshold_pos = 12.0f;

				bool similarPos = diff_pos < threshold_pos;
				bool similarAngle = diff_angle < threshold_radians;
				bool isLarge = length(basisvector1) > 8.0f || length(basisvector2) > 8.0f;
				
				if(warp.thread_rank() % 2 == 1){
					isRejected = !isLarge && similarPos && similarAngle;
				}
			}
			// if(iteration > 10 && index % 2 == 0) isRejected = true;

			bool isAccepted = !isRejected;

			// compute 32 bit mask where each thread reports whether it is discarding its splat
			uint32_t acceptance = isAccepted ? 1 : 0;
			uint32_t acceptanceMask = warp.ballot(acceptance);

			if(warp.thread_rank() == 0){
				uint32_t warpRank = block.thread_rank() / 32;
				sh_acceptanceMasks[warpRank] = acceptanceMask;
			}

			#ifdef DEBUGING
			if(isRejected){
				atomicAdd(&numRejected, 1);
			}
			#endif
		}

		block.sync();

		__shared__ int splatsInSharedMemory;

		int splatsInIteration = min(numPointsInTile - PREFETCH_COUNT * iteration, PREFETCH_COUNT);

		// { // single-threaded
		// 	if(block.thread_rank() == 0){
		// 		splatsInSharedMemory = 0;

		// 		for(int i = 0; i < splatsInIteration; i++){

		// 			// if(i % 32 == 0){
		// 			// 	acceptanceMask = sh_acceptanceMasks[i / 32];
		// 			// }

		// 			uint32_t acceptanceMask = sh_acceptanceMasks[i / 32];

		// 			bool accepted = (acceptanceMask >> (i % 32)) & 1;
		// 			// bool accepted = acceptanceMask & 1;
		// 			// acceptanceMask = acceptanceMask >> 1;

		// 			if(accepted){
		// 				sh_rasterdata_coalesced[splatsInSharedMemory] = sh_rasterdata[i];
		// 				splatsInSharedMemory++;
		// 			}
		// 		}
		// 	}
		// }

		{ // multi-threaded

			// compute per-warp prefix-sum of the number of accepted splats
			if(block.thread_rank() < PREFETCH_COUNT){
				uint32_t warpRank = block.thread_rank() / 32;
				uint32_t laneID = block.thread_rank() % 32;
				uint32_t acceptanceMask = sh_acceptanceMasks[warpRank];
				uint32_t acceptance = min((acceptanceMask >> laneID) & 1, 1);
				uint32_t acceptanceSubgroup = __ballot_sync(0xffffffff, acceptance == 1);
				uint32_t numAccepted = (uint32_t) __popc(acceptanceSubgroup);

				if(laneID == 0){
					sh_numAccepted[warpRank] = numAccepted;
				}
			}

			block.sync();

			if(block.thread_rank() == 0){
				sh_prefixSum[0] = 0;
				sh_prefixSum[1] = sh_prefixSum[0] + sh_numAccepted[0];
				sh_prefixSum[2] = sh_prefixSum[1] + sh_numAccepted[1];
				sh_prefixSum[3] = sh_prefixSum[2] + sh_numAccepted[2];

				splatsInSharedMemory = sh_prefixSum[3] + sh_numAccepted[3];
			}

			block.sync();

			// Now pack the accepted splats together into consecutive shared memory
			if(block.thread_rank() < PREFETCH_COUNT)
			{
				auto warp = cg::coalesced_threads();
				uint32_t warpRank = block.thread_rank() / 32;
				uint32_t laneID = block.thread_rank() % 32;

				uint32_t acceptanceMask = sh_acceptanceMasks[warpRank];
				uint32_t acceptance = min((acceptanceMask >> laneID) & 1, 1);

				uint32_t prefixSum_warp = exclusive_scan(warp, acceptance);

				if(acceptance){
					sh_rasterdata_coalesced[sh_prefixSum[warpRank] + prefixSum_warp] = sh_rasterdata[block.thread_rank()];
				}
			}
		}

		block.sync();

		// iterate with all threads of block through accepted splats
		// i.e., all 256 threads process first splat, then all 256 threads process second, ...
		// uint64_t t_start_draw = nanotime();
		for(int i = 0; i < splatsInSharedMemory; i++)
		{
			auto rasterdata = sh_rasterdata_coalesced[i];
			vec4 splatColor = rasterdata.colors;
			vec2 imgCoords = rasterdata.imgpos;

			vec2 pFrag = {fpixel_x - imgCoords.x, fpixel_y - imgCoords.y};

			float sT = dot(rasterdata.dir1, pFrag); 
			float sB = dot(rasterdata.dir2, pFrag); 
			float w = (sT * sT + sB * sB);

			if(rasterdata.depth > depth) break;
			if(pixel_x >= width || pixel_y >= height) break;
			
			// Splat boundary at w = 1.0. The remaining part of the gaussian is discarded.
			if(w < 1.0f)
			{
				// vec4 splatColor = rasterdata.colors;

				w = exp(-4.0f * w);
				
				remainingTranslucency = 1.0f - pixel.w;

				if(args.uniforms.showRing && w < 0.05f){
					splatColor.a = 1.0f;
					w = 1.0f;
				}

				float alpha = w * remainingTranslucency * splatColor.a;

				pixel.r += alpha * splatColor.r;
				pixel.g += alpha * splatColor.g;
				pixel.b += alpha * splatColor.b;
				pixel.w += alpha;
				
				// update depth buffer
				// if(remainingTranslucency < 0.030f)
				if(remainingTranslucency < 0.030f && (rasterdata.flags & FLAGS_DISABLE_DEPTHWRITE) == 0)
				{
					depth = rasterdata.depth;
					uint64_t udepth = __float_as_uint(depth);
					uint64_t pixel = (udepth << 32) | 0x00000000;
					
					target.framebuffer[pixelID] = pixel;
					
					break;
				}
			}
		}

		sh_remainingTranslucency[block.thread_rank()] = remainingTranslucency;

		block.sync();
	}

	block.sync();

	// if(isDebugTile(tile_x, tile_y)){
	// 	args.state->dbg.numAccepted = dbg_numAccepted;
	// 	args.state->dbg.numRejected = dbg_numRejected;
	// }

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

	#ifdef DEBUGING

	if(isDebugTile(tile_x, tile_y)){
		if(block.thread_rank() == 0){
			
			printf("numRejeced: %d out of %d \n", numRejected, numPointsInTile);
		}
	}

	#endif
}
