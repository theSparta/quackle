#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/platform/env.h"
#include <unordered_set>
using namespace tensorflow;

const int FEATURE_SIZE = 28;
const int BOARD_SIZE = 15;
typedef std::vector<std::vector<char> > Feature;
typedef std::vector<std::vector<std::vector<short> > > Feature_Image;

class NNInference
{
private:
	std::string graph_definition;
	int feature_size;
	Session* session;
	std::vector<Tensor> outputs;
	Tensor input_tensor, input_tensor2;
	Feature_Image curr_state, next_state;
	const std::unordered_set<char> blanks = {' ', '"', '\'', '^', '=', '~', '-'};
	const std::vector<int> tile_values = {1, 3, 3, 2, 1, 4, 2, 4, 1, 8,
		5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10};
public:
	NNInference(const string &);
	~NNInference();
	float getOutput(const Feature & curr_state, const Feature & next_state, std::vector<float> &);
	void convert(const Feature &, Feature_Image & );
};