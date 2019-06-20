/* Copyright (c) 2019 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <gtest/gtest.h>
#include "alsutils/io/TextFileCache.hpp"
#include <boost/filesystem.hpp>
#include <fstream>
#include <streambuf>

// Test that the files show up in loaded files
TEST(TextFileCacheTest, TestGetLoadedFiles) {
    std::vector<std::string> filesToWrite = {
        "deleteme_TextFileCacheTest_TestGetLoadedFiles_file1.txt",
        "deleteme_TextFileCacheTest_TestGetLoadedFiles_file2.txt"
    };

    auto fileContent = [](const std::string & filename) {
        return "In file " + filename + "\n";
    };

    for (auto filename : filesToWrite) {
        std::ofstream file(filename);

        file << fileContent(filename);
    }

    for (auto filename : filesToWrite) {
        auto& instance = alsutils::io::TextFileCache::getInstance();
        auto content = instance.loadTextFile(filename);

        const auto expectedContent = fileContent(filename);

        ASSERT_EQ(expectedContent, content);
    }

    for (auto filename : filesToWrite) {
        auto& instance = alsutils::io::TextFileCache::getInstance();


        bool found = false;

        for (auto loadedFilename : instance.getAllLoadedFiles()) {
            if (loadedFilename == boost::filesystem::absolute(filename)) {
                found = true;
            }
        }

        ASSERT_TRUE(found);

    }
}


TEST(TextFileCacheTest, TestCachingFiles) {
    std::vector<std::string> filesToWrite = {
        "deleteme_TextFileCacheTest_TestCachingFiles_file1.txt",
        "deleteme_TextFileCacheTest_TestCachingFiles_file2.txt"
    };

    auto fileContent = [](const std::string & filename) {
        return "In file cached " + filename + "\n";
    };

    for (auto filename : filesToWrite) {
        std::ofstream file(filename);

        file << fileContent(filename);
    }

    for (auto filename : filesToWrite) {
        auto& instance = alsutils::io::TextFileCache::getInstance();
        auto content = instance.loadTextFile(filename);

        const auto expectedContent = fileContent(filename);

        ASSERT_EQ(expectedContent, content);
    }

    for (auto filename : filesToWrite) {
        std::ofstream file(filename);

        file << "Not cached in " << filename << "\n";
    }


    for (auto filename : filesToWrite) {
        auto& instance = alsutils::io::TextFileCache::getInstance();
        auto content = instance.loadTextFile(filename);

        const auto expectedContent = fileContent(filename);

        ASSERT_EQ(expectedContent, content);
    }

}


