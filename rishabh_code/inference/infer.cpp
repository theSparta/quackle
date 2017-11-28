#include "infer.h"
#include <assert.h>

NNInference::NNInference(const string & weights_file)
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
	input_tensor = Tensor(DT_FLOAT, TensorShape({1, BOARD_SIZE, BOARD_SIZE, 2*FEATURE_SIZE}));
	input_tensor2 = Tensor(DT_FLOAT, TensorShape({1, 2}));
	curr_state = Feature_Image(FEATURE_SIZE,
		std::vector<std::vector<short> >(BOARD_SIZE, std::vector<short>(BOARD_SIZE, 0)));
	next_state = Feature_Image(FEATURE_SIZE,
		std::vector<std::vector<short> >(BOARD_SIZE, std::vector<short>(BOARD_SIZE, 0)));
}

float NNInference::getOutput(const Feature & s1, const Feature & s2, std::vector<float> & features)
{
	outputs.clear();
	convert(s1, curr_state);
	convert(s2, next_state);
	auto tensor_map = input_tensor.tensor<float, 4>();
	for(int k = 0; k < FEATURE_SIZE ; k++)
		for(int i = 0; i < BOARD_SIZE; i++)
			for (int j = 0; j < BOARD_SIZE; ++j){
				tensor_map(0, i, j, k) = curr_state[k][i][j];
				tensor_map(0, i, j, k + FEATURE_SIZE) = next_state[k][i][j];
			}
	std::copy_n(features.begin(), 2, input_tensor2.flat<float>().data());
	TF_CHECK_OK(session->Run({{"input1", input_tensor}, {"input2", input_tensor2}},
		{"output_node0"}, {}, &outputs));
	float output = outputs[0].scalar<float>()(0);
	return output;
}


NNInference::~NNInference()
{
	session->Close();
	delete session;
}

void NNInference::convert(const Feature & s, Feature_Image & image)
{
	int index, small_index;
	for(int i = 0; i < s.size(); i++){
		for(int j = 0; j < s[i].size(); j++){
			for(int k = 0; k < image.size(); k++)
				image[k][i][j] = 0;
			if(blanks.count(s[i][j])){
				image[0][i][j] = 1;
				if(s[i][j] != ' ')
					image[FEATURE_SIZE - 1][i][j] = -1;
			}
			else{
				index = (s[i][j] - 'A') + 1;
				if(1 <= index && index <= 26){
					image[index][i][j] = 1;
					image[FEATURE_SIZE - 1][i][j] = tile_values[index-1];
				}
				else{
					small_index = (s[i][j] - 'a') + 1;
					if(1 <= small_index && small_index <= 26){
						image[small_index][i][j] = 1;
					}
				}
			}
		}
	}
}