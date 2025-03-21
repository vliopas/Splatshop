// Track Actions to undo/redo them.

// #include <vector>

// #include "cuda.h"
// #include "cuda_runtime.h"

// #include "HostDeviceInterface.h"

// using namespace std;

// struct Action{

// 	virtual void undo() = 0;
// 	virtual void apply() = 0;

// };

// struct ActionHistory{

// 	vector<Action> history;

// };

// struct ActionSelect{
// 	GaussianData model;
// 	CUdeviceptr cptr_indices;

// 	void apply(){

// 	}
// };