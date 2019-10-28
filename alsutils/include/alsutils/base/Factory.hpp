#pragma once
#include "alsutils/base/impl/FactoryRegistry.hpp"
namespace alsutils {
namespace base {

//! This is the base class for all "automatic factories"
//!
//! The way this class is meant to be used, is to subclass this class
//! specializing the templates
//!
//! Example
//! \code{.cpp}
//! class A {
//! public:
//!    virtual void f() = 0;
//!     static std::string getClassName() { return "namespace::A"; }
//! };
//!
//! class AFactory : public alsutils::base::Factory<A, const std::string&>
//!
//! class B : public A {
//! public:
//!    B(const std::string & a) : a(a) {}
//!    virtual void f() override {
//!       std::cout << a << std::endl;
//!    }
//!
//! private:
//!     const static bool initialized;
//! };
//!
//! // This is usually in the .cpp file
//! const bool B::initialized = AFactory::registerClassWithOnlyConstructor("B");
//! \endcode
//!
//!
template<class T, class ... Args>
class Factory {
public:

    typedef typename impl::FactoryRegistry<T, Args...>::PointerType PointerType;
    typedef typename impl::FactoryRegistry<T, Args...>::CreatorType CreatorType;

    //! Creates a new instance of the class with the given name and arguments.
    static PointerType createInstance(const std::string& name, Args... args) {
        return impl::FactoryRegistry<T, Args...>::getInstance().createInstance(name,
                args...);
    }

    //! Registers a new class to the registry.
    //! @param name the name of the class
    //! @param creator the creator/constructor
    static bool registerClass(const std::string& name,
        CreatorType creator) {
        return impl::FactoryRegistry<T, Args...>::getInstance().registerClass(name,
                creator);
    }


    //! Registers a new class to the registry, in this case,
    //! we assume the class has a constructor that takes all the default arguments
    //! @param name the name of the class
    template<class C>
    static bool registerClassWithOnlyConstructor(const std::string& name) {
        return impl::FactoryRegistry<T, Args...>::getInstance().template
            registerClassWithOnlyConstructor<C>(name);
    }
};
} // namespace base
} // namespace alsutils
