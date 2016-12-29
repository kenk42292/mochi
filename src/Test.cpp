#include "cute.h"
#include "ide_listener.h"
#include "xml_listener.h"
#include "cute_runner.h"

// TODO #include the headers for the code you want to test
#include "../test/VanillaFeedForwardTest.hpp"

// TODO Add your test functions

bool runAllTests(int argc, char const *argv[]) {
    cute::suite s { };

    //TODO add your test here
    s.push_back(CUTE(VanillaFeedForwardTest::feedForwardTest1));
    s.push_back(CUTE(VanillaFeedForwardTest::feedForwardTest2));
    cute::xml_file_opener xmlfile(argc, argv);
    cute::xml_listener<cute::ide_listener<>> lis(xmlfile.out);
    auto runner = cute::makeRunner(lis, argc, argv);
    bool success = runner(s, "AllTests");
    return success;
}

int main(int argc, char const *argv[]) {
    return runAllTests(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
