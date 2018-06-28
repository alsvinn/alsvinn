#pragma once
#include <memory>
#include <functional>
#include <string>
#include <map>
#include "alsutils/error/Exception.hpp"


namespace alsutils {
namespace base {
namespace impl {

//! T is any class that has the method
//!
//! \code{.cpp}
//! static std::string getClassName() {
//!    // should return the class name
//! }
//! \endcode
//!
//! \note this class is a singleton.
template<class T, class ... Args>
class FactoryRegistry {
private:
    // singleton and non-copyable
    FactoryRegistry() {}
    FactoryRegistry(const FactoryRegistry& other) {}
public:

    typedef std::shared_ptr<T> PointerType;
    typedef std::function<PointerType(Args...)> CreatorType;

    //! Registers a new class to the registry.
    //! @param name the name of the class
    //! @param creator the creator/constructor
    bool registerClass(const std::string& name,
        CreatorType creator) {
        if (creators.find(name) != creators.end()) {
            THROW("Class named " << name << " for base class " <<
                T::getClassName() << " already registered with the factory.");
        }

        creators[name] = creator;
        return true;
    }

    //! Registers a new class to the registry, in this case,
    //! we assume the class has a constructor that takes all the default arguments
    //! @param name the name of the class
    template<class C>
    bool registerClassWithOnlyConstructor(const std::string& name) {
        auto constructor = [&](Args... args) {
            PointerType pointer;
            pointer.reset(new C(args...));
            return pointer;
        };
        return registerClass(name, constructor);
    }


    //! Creates a new instance of the class with the given name and arguments.
    PointerType createInstance(const std::string& name, Args... args) {
        if (creators.find(name) == creators.end()) {
            THROW("Could not find class named " << name << " for base class " <<
                T::getClassName());
        }

        return creators[name](args...);
    }

    //! Gets the singleton instance of this class
    static FactoryRegistry& getInstance() {
        static FactoryRegistry factory;
        return factory;
    }
private:
    std::map<std::string, CreatorType> creators;
};
} // namespace impl
} // namespace base
} // namespace alsutils
