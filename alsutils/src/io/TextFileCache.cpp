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

#include "alsutils/io/TextFileCache.hpp"
#include "alsutils/error/Exception.hpp"
#include <fstream>
#include <boost/filesystem.hpp>
#include <streambuf>
#include <sstream>
#include <mutex>


namespace alsutils {
namespace io {

TextFileCache& TextFileCache::getInstance() {
    // Meyer singleton
    std::lock_guard<std::recursive_mutex> lock(mutex);
    static TextFileCache instance;
    return instance;

}

std::string TextFileCache::loadTextFile(const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto absolutePath = boost::filesystem::absolute(path).string();
    auto inMap = loadedTextFiles.find(absolutePath);

    if (inMap != loadedTextFiles.end()) {
        return inMap->second;
    }

    std::ifstream file(absolutePath);

    if (!file.good()) {
        THROW("Could not open file " << absolutePath);
    }

    std::string text((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    loadedTextFiles[absolutePath] = text;

    return text;
}


std::vector<std::string> TextFileCache::getAllLoadedFiles() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::vector<std::string> loadedFilenames;

    for (auto keyValue : loadedTextFiles) {
        loadedFilenames.push_back(keyValue.first);
    }

    return loadedFilenames;
}

std::recursive_mutex TextFileCache::mutex;

}
}
