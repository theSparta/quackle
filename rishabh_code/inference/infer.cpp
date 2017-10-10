#include "infer.h"

NNInference::NNInference(const string & weights_file, int feature_size)
{
	// assert(weights_file.en)
	graph_definition = weights_file;
	GraphDef graph_def;
	SessionOptions opts;
	TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));

	graph::SetDefaultDevice("/cpu:0", &graph_def);
	// opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
	// opts.config.mutable_gpu_options()->set_allow_growth(true);

	// create a new session
	TF_CHECK_OK(NewSession(opts, &session));

	// Load graph into session
	TF_CHECK_OK(session->Create(graph_def));
	input_tensor = Tensor(DT_FLOAT, TensorShape({1, feature_size}));
}

float NNInference::getOutput(std::vector<float> & features)
{
	outputs.clear();
	std::copy_n(features.begin(), features.size(), input_tensor.flat<float>().data());
	TF_CHECK_OK(session->Run({{"x", input_tensor}}, {"output_node0"}, {}, &outputs));
	float output = outputs[0].scalar<float>()(0);
	return output;
}

NNInference::~NNInference()
{
	session->Close();
	delete session;
}