#include <Functions/FunctionFactory.h>
#include <Functions/FunctionBinaryArithmetic.h>



namespace DB
{
namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

namespace
{

template <typename A, typename B>
struct LookupImpl
{
    using ResultType = typename NumberTraits::ResultOfBit<A, B>::Type;
    static constexpr bool allow_fixed_string = true;

    template <typename Result = ResultType>
    static inline Result apply(A a, B b)
    {
        /// The results of bitcount(xor(a,b)) need to be pre-stored into the lookuptable.
        /// Bitcount(a) obeys the similar rule.
        return lookuptable[static_cast<Result>(a), static_cast<Result>(b)];
    }

#if USE_EMBEDDED_COMPILER
    static constexpr bool compilable = true;

    static inline llvm::Value * compile(llvm::IRBuilder<> & b, llvm::Value * left, llvm::Value * right, bool)
    {
        if (!left->getType()->isIntegerTy())
            throw Exception("LookupImpl expected an integral type", ErrorCodes::LOGICAL_ERROR);
        return b.CreateXor(left, right);
    }
#endif
};

struct NameLookup { static constexpr auto name = "lookup"; };
using FunctionLookup = BinaryArithmeticOverloadResolver<LookupImpl, NameLookup, true, false>;

}

void registerFunctionLookup(FunctionFactory & factory)
{
    factory.registerFunction<FunctionLookup>();
}

}
