


extern "C" __global__
void kernel_render_gaussians(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, StageData* stagedatas
){

	auto grid     = cg::this_grid();
	auto block    = cg::this_thread_block();

	int width     = target.width;
	int height    = target.height;
	int tiles_x   = int(width + TILE_SIZE_3DGS - 1) / int(TILE_SIZE_3DGS);
	int tiles_y   = int(height + TILE_SIZE_3DGS - 1) / int(TILE_SIZE_3DGS);
	int tileID = blockIdx.x;

	// tile coordinates
	int tile_x = tileID % tiles_x;
	int tile_y = tileID / tiles_x;

	int dx = 30;
	int dy = 20;

	// dx = 0; dy = 0;

	bool outsideX = tile_x < dx || tile_x > tiles_x - dx;
	bool outsideY = tile_y < dy || tile_y > tiles_y - dy;
	// if(outsideX || outsideY){
		// kernel_render_gaussians_lq(args, target);
	// }else{
		kernel_render_gaussians_hq(args, target, tiles, indices, stagedatas);
	// }
		// kernel_render_gaussians_lq(args, target);

}