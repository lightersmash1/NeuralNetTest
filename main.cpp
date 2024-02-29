#include <Windows.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <map>
#include <numeric>

using namespace std;

typedef long double ufloat; // very precise (insert a precise number type here)

ufloat relu(ufloat x) { // leaky relu (help me god please)
	return tanh(x); // replace with something that isn't between -1 and 1
}
ufloat relud(ufloat x) { // leaky relu derivative (help me god please)
	return 1 - pow(tanh(x), 2);
}

ufloat mean(vector<ufloat> abc) {
	ufloat abcd = 0;
	for (int i = 0; i < abc.size(); i++) {
		abcd += abc[i];
	}
	return abcd / abc.size();
}

class LinearNode {
public:

	ufloat weight = 0;
	ufloat bias = 0;
	LinearNode(ufloat weight2 = 0, ufloat bias2 = 0) {
		if (weight2 == 0 && bias2 == 0) {
			weight = (ufloat)(rand() % 1000) / 1000;
			bias = (ufloat)(rand() % 1000) / 1000;
			return;
		}
		weight = weight2;
		bias = bias2;
	}

	ufloat compute(ufloat input) {
		return (weight * input) + bias;
	}
};

vector<ufloat> CalculateAIModel(vector<vector<vector<LinearNode>>> AIModel, vector<ufloat> Input) { // theres an issue hidden in here
	vector<ufloat> lastresult = Input;
	vector<ufloat> newresult = Input;
	for (int i = 0; i < AIModel.size(); i++) {
		vector<vector<LinearNode>> layer = AIModel[i];
		for (int resultnumber = 0; resultnumber < layer.size(); resultnumber++) {
			ufloat val = 0;
			vector<LinearNode> weights = layer[resultnumber];
			int siz = weights.size();
			for (int wow = 0; wow < siz; wow++) {
				val += weights[wow].compute(lastresult[wow]);
			}
			newresult[resultnumber] = relu(val / siz); // can replace relu(
		}
		lastresult = newresult; // push back
	}
	return lastresult;
}

// constants for correction

ufloat multiplier = 0.01; // this is used for the biases

ufloat loss(ufloat predicted, ufloat actual) { // Loss Function (google it)
	return (actual - predicted); // thanks wikipedia https://wikimedia.org/api/rest_v1/media/math/render/svg/124390fc1208653d027808db415897fa19d0ab33
}

// just in case i never finish this: if you ever come up with a good back propagation idea \/ add it

pair<vector<vector<vector<LinearNode>>>, ufloat> BackPropagateAIModel(vector<vector<vector<LinearNode>>> AIModel, vector<ufloat> Output, vector<ufloat> TrainingData) {
	vector<vector<vector<LinearNode>>> newAIModel = AIModel;
	vector<ufloat> error;
	for (int i = 0; i < Output.size(); i++) {
		error.push_back(loss(relud(Output[i]), TrainingData[i]));
	}
	for (int i = 0; i < AIModel.size(); i++) {
		vector<vector<LinearNode>> layer = AIModel[i];
		for (int j = 0; j < layer.size(); j++) {
			vector<LinearNode> largernode = layer[j];
			ufloat layermultiplier = multiplier * (i + 1);
			for (int k = 0; k < largernode.size(); k++) {
				ufloat err = error[k] * layermultiplier;
				newAIModel[i][j][k].bias += err;
				newAIModel[i][j][k].weight += err;
			}
		}
	}
	return {newAIModel, mean(error)};
}

int main() {
	srand(time(NULL));
	vector<vector<vector<LinearNode>>> AIModel = { {{LinearNode(), LinearNode()}}, {{LinearNode()}} };
	vector<vector<vector<ufloat>>> TrainingModels = {
		{{2}, {18}},
		{{3}, {20}},
		{{4}, {17}},
		{{3}, {18}},
		{{1}, {12}},
		{{4}, {19}},
		{{20}, {4}},
		{{19}, {7}},
		{{18}, {5}},
		{{20}, {6}},
		{{19}, {4}},
		{{18}, {3}},
		{{20}, {18}},
		{{19}, {20}},
		{{18}, {17}},
	};
	// epoch: how many times you loop through the Training
	int epochs = 1000;
	for (int l = 0; l < epochs; l++) {
		for (int i = 0; i < TrainingModels.size(); i++) {
			auto lol = BackPropagateAIModel(AIModel, CalculateAIModel(AIModel, TrainingModels[i][0]), TrainingModels[i][1]);
			AIModel = lol.first;
		}
	}
	for (int i = 0; i < TrainingModels.size(); i++) {
		auto lol = BackPropagateAIModel(AIModel, CalculateAIModel(AIModel, TrainingModels[i][0]), TrainingModels[i][1]);
		AIModel = lol.first;
		cout << lol.second << endl;
	}
	for (int i = 0; i < 50; i++) {
		cout << "(" << i << ", ";
		cout << CalculateAIModel(AIModel, {(ufloat)i})[0] << ")" << endl;
		//AIModel = BackPropagateAIModel(AIModel, CalculateAIModel(AIModel, TrainingModels[i][0]), TrainingModels[i][1]);
	}
	//cout << LinearNode(AIModel[0][0][0].weight, AIModel[0][0][0].bias).compute(4) << endl;
	return 0;
}
