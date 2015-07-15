#pragma once

namespace alsfvm {
	namespace memory {
		///
		/// The baseclass for all memory objects, this is 
		/// untemplated so it can be passed around easier.
		///
		class MemoryBase
		{
		public:
			MemoryBase() {}
			virtual ~MemoryBase() {}
		};

	}
}