

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
	int tileID    = blockIdx.x;
	Tile tile     = tiles[tileID];

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

	int numPointsInTile = tile.lastIndex - tile.firstIndex + 1;
	if(tile.firstIndex > tile.lastIndex) {
		numPointsInTile = 0;
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
		color = 0xff0000ff;

		uint32_t C = target.framebuffer[pixelID] & 0xffffffff;
		vec4 oldPixel = vec4{
			(C >>  0) & 0xff,
			(C >>  8) & 0xff,
			(C >> 16) & 0xff,
			(C >> 24) & 0xff,
		};

		rgba[3] = 255;

		// int cindex = numProcessed / 1000.0f;
		int cindex = log10(float(numPointsInTile)) / 2.0f;
		cindex = pow(float(numPointsInTile), 0.5f) * 0.05f;
		// cindex = pow(float(30'000), 0.5f) * 0.05f;
		cindex = clamp(cindex, 0, 10);

		cindex = 0;
		if(numPointsInTile < 10) cindex = 1;
		else if(numPointsInTile <   100) cindex = 2;
		else if(numPointsInTile <  1000) cindex = 3;
		else if(numPointsInTile <  2000) cindex = 4;
		else if(numPointsInTile <  4000) cindex = 5;
		else if(numPointsInTile <  8000) cindex = 6;
		else if(numPointsInTile < 16000) cindex = 7;
		else if(numPointsInTile < 32000) cindex = 8;
		else if(numPointsInTile < 64000) cindex = 9;

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