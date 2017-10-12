#include "move_create.h"
#include "../test/trademarkedboards.h"

Quackle::Game* Move_Create::createNewGame(const string & gameFile)
{
    QuackleIO::GCGIO io;
    QString filename = QuackleIO::Util::uvStringToQString(gameFile);
    QFile file(filename);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        UVcout << "Could not open gcg " << QuackleIO::Util::qstringToString(filename) << endl;
        return 0;
    }

    QTextStream in(&file);
    Quackle::Game *game = io.read(in, QuackleIO::Logania::MaintainBoardPreparation);
    file.close();

    return game;
}

void Move_Create::getMove(const string & s1, const string & s2)
{
	Quackle::Move m;
	m = Quackle::Move::createPlaceMove(s1, QUACKLE_ALPHABET_PARAMETERS->encode(s2));
	cout << m.effectiveScore() << endl;
	UVcout << m.toString() << endl;
	UVcout << currPosition << endl;

}

void Move_Create::init()
{
	m_dataManager.setBackupLexicon("twl06");
	m_dataManager.setAppDataDirectory("../data");

	QString alphabetFile = QuackleIO::Util::stdStringToQString(Quackle::AlphabetParameters::findAlphabetFile("english"));
	QuackleIO::FlexibleAlphabetParameters *flexure = new QuackleIO::FlexibleAlphabetParameters;
	if (flexure->load(alphabetFile))
	{
		m_dataManager.setAlphabetParameters(flexure);
	}
	else
	{
		UVcerr << "Couldn't load alphabet english" << endl;
		delete flexure;
	}
	m_dataManager.setBoardParameters(new ScrabbleBoard());
   	m_dataManager.lexiconParameters()->loadDawg(Quackle::LexiconParameters::findDictionaryFile("csw12.dawg"));
   	m_dataManager.lexiconParameters()->loadGaddag(Quackle::LexiconParameters::findDictionaryFile("csw12.gaddag"));
   	m_dataManager.strategyParameters()->initialize("csw12");
	evaluator = new Quackle::ModifiedEvaluator(0);
}

void Move_Create::setGame(const string & gameFile)
{
	delete game;
	Quackle::Game *game = createNewGame(gameFile);
	if(game)
	{
		currPosition = game->currentPosition();
	}
}

vector<float> Move_Create::getFeatures(const string & s1, const string & s2)
{
	Quackle::Move move;
	if (s1.empty()){
		move = Quackle::Move::createExchangeMove(
			QUACKLE_ALPHABET_PARAMETERS->encode(s2), false);
	}
	else{
		move = Quackle::Move::createPlaceMove(s1,
			QUACKLE_ALPHABET_PARAMETERS->encode(s2));
	}
	currPosition.scoreMove(move);
	return evaluator->getFeatures(currPosition, move);
}

Quackle::DataManager Move_Create::m_dataManager;
Quackle::ModifiedEvaluator* Move_Create::evaluator;

int main()
{
	Move_Create::init();
	Move_Create x;
	x.setGame("../test/gcg/Five_Minute_Championship_Player-game-714#8#4.gcg");
	// x.getMove("", "FNNOQ");
	auto y = x.getFeatures("", "FNNOQ");
	for (auto &i : y)
		cout << i << " ";
	cout << endl;
	return 0;
}
