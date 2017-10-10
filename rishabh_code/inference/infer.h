#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/platform/env.h"
using namespace tensorflow;

class NNInference
{
private:
	std::string graph_definition;
	Session* session;
	std::vector<Tensor> outputs;
	Tensor input_tensor;
public:
	NNInference(const string &, int);
	~NNInference();
	float getOutput(std::vector<float> &);
};