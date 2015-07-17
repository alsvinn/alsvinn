#include "gtest/gtest.h"
#include <vector>
#include "alsfvm/io/io_utils.hpp"

TEST(IOTest, NamingTest) {
    std::string basename("name");

    const size_t snapshotNumber = 10;

    auto generatedName = alsfvm::io::getOutputname(basename, snapshotNumber);

    // We only require that these two tings should be in the name
    ASSERT_TRUE(generatedName.find(basename) !=std::string::npos);

    ASSERT_TRUE(generatedName.find("10") !=std::string::npos);
}
