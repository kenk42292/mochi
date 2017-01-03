#include "cute.h"
#include "ide_listener.h"
#include "xml_listener.h"
#include "cute_runner.h"

// TODO #include the headers for the code you want to test
#include "../test/VanillaFeedForwardTest.hpp"
#include "../test/SigmoidTest.hpp"
#include "../test/SoftplusTest.hpp"
#include "../test/MaxPoolTest.hpp"

// TODO Add your test functions

bool runAllTests(int argc, char const *argv[]) {
    cute::suite s { };

    //TODO add your test here
    s.push_back(CUTE(VanillaFeedForwardTest::feedForwardTest1));
    s.push_back(CUTE(VanillaFeedForwardTest::feedForwardTest2));
    s.push_back(CUTE(VanillaFeedForwardTest::backPropTest1));
    s.push_back(CUTE(SigmoidTest::feedForwardTest1));
    s.push_back(CUTE(SigmoidTest::feedForwardTest2));
    s.push_back(CUTE(SigmoidTest::backPropTest1));
    s.push_back(CUTE(SigmoidTest::backPropTest2));
    s.push_back(CUTE(SoftplusTest::feedForwardTest1));
    s.push_back(CUTE(SoftplusTest::feedForwardTest2));
    s.push_back(CUTE(SoftplusTest::backPropTest1));
    s.push_back(CUTE(SoftplusTest::backPropTest2));
    s.push_back(CUTE(MaxPoolTest::feedForwardTest1));
    s.push_back(CUTE(MaxPoolTest::backPropTest1));
    cute::xml_file_opener xmlfile(argc, argv);
    cute::xml_listener<cute::ide_listener<>> lis(xmlfile.out);
    auto runner = cute::makeRunner(lis, argc, argv);
    bool success = runner(s, "AllTests");
    return success;
}

int main(int argc, char const *argv[]) {
    return runAllTests(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
