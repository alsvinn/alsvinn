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

#pragma once
#include <string>
#include <map>
#include <vector>
#include <thread>
#include <mutex>

namespace alsutils {
namespace io {

//! Simple file cache. This is mainly use to reduce IO, but also to store
//! the loaded .xml and .py files for later reproducibility
//!
//! Usage
//! \code{.cpp}
//! auto instance = TextFileCache::getInstance();
//! auto text = instance.loadTextFile("some_file.xml");
//!
//! // Can also loop through all loaded text files
//! for (auto filename : instance.getAllLoadedFiles()) {
//!     auto previouslyLoadedText = instance.loadTextFile(filename);
//! }
//! \endcode
class TextFileCache {
public:
    TextFileCache(const TextFileCache&) = delete;

    static TextFileCache& getInstance();

    std::string loadTextFile(const std::string& path);

    std::vector<std::string> getAllLoadedFiles() const;


private:
    TextFileCache() = default;

    std::map<std::string, std::string> loadedTextFiles;

    static std::recursive_mutex mutex;

};
} // namespace io
} // namespace alsutils
