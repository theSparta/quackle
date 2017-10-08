#include <QtCore>
#include <QString>
// #include <QS
#include <iostream>
#include <string>

#include "boardparameters.h"
#include "computerplayer.h"
#include "datamanager.h"
#include "generator.h"
#include "lexiconparameters.h"
#include "strategyparameters.h"
#include "game.h"

// #include "quackleio/dictimplementation.h"
#include "quackleio/flexiblealphabet.h"
// #include "quackleio/froggetopt.h"
#include "quackleio/gcgio.h"
#include "quackleio/util.h"

class Move_Create {
private:
    Quackle::GamePosition currPosition;
    Quackle::Game *game;
    Quackle::Game* createNewGame(const string & filename);
public:
	static Quackle::DataManager m_dataManager;
    static Quackle::ModifiedEvaluator* evaluator;
    void getMove(const string & s1, const string & s2);
    static void init();
    void setGame(const string & filename);
    vector<float> getFeatures(const string & s1, const string & s2);
};
