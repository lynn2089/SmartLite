#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionSumBin.h>
#include <AggregateFunctions/Helpers.h>
#include <AggregateFunctions/FactoryHelpers.h>
#include <common/extended_types.h>
#include <common/defines.h>


namespace DB
{
struct Settings;

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
}

namespace
{
bool allowType(const DataTypePtr& type) noexcept
{
    const WhichDataType t(type);
    return t.isInt() || t.isUInt();
}

template <typename T>
struct SumBinSimple
{
    using ResultType = T;
    using AggregateDataType = AggregateFunctionSumBinData<ResultType>;
    using Function = AggregateFunctionSumBin<T, ResultType, AggregateDataType, AggregateFunctionTypeSumBin>;
};

template <typename T>
struct SumBinSameType
{
    using ResultType = T;
    using AggregateDataType = AggregateFunctionSumBinData<ResultType>;
    using Function = AggregateFunctionSumBin<T, ResultType, AggregateDataType, AggregateFunctionTypeSumBinWithOverflow>;
};

template <typename T>
struct SumBinKahan
{
    using ResultType = Float64;
    using AggregateDataType = AggregateFunctionSumBinKahanData<ResultType>;
    using Function = AggregateFunctionSumBin<T, ResultType, AggregateDataType, AggregateFunctionTypeSumBinKahan>;
};

template <typename T> using AggregateFunctionSumBinSimple = typename SumBinSimple<T>::Function;
template <typename T> using AggregateFunctionSumBinWithOverflow = typename SumBinSameType<T>::Function;
template <typename T> using AggregateFunctionSumBinKahan =
    std::conditional_t<IsDecimalNumber<T>, typename SumBinSimple<T>::Function, typename SumBinKahan<T>::Function>;


template <template <typename> class Function>
AggregateFunctionPtr createAggregateFunctionSumBin(const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    assertNoParameters(name, parameters);
    assertUnary(name, argument_types);

    AggregateFunctionPtr res;
    DataTypePtr data_type = argument_types[0];
    if (!allowType(data_type))
        throw Exception("Illegal type " + data_type->getName() + " of argument for aggregate function " + name,
            ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
            
    if (isDecimal(data_type))
        res.reset(createWithDecimalType<Function>(*data_type, *data_type, argument_types));
    else
        res.reset(createWithNumericType<Function>(*data_type, argument_types));

    if (!res)
        throw Exception("Illegal type " + argument_types[0]->getName() + " of argument for aggregate function " + name,
                        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}

}

void registerAggregateFunctionSumBin(AggregateFunctionFactory & factory)
{
    factory.registerFunction("sumbin", createAggregateFunctionSumBin<AggregateFunctionSumBinSimple>, AggregateFunctionFactory::CaseInsensitive);
    factory.registerFunction("sumbinWithOverflow", createAggregateFunctionSumBin<AggregateFunctionSumBinWithOverflow>);
    factory.registerFunction("sumbinKahan", createAggregateFunctionSumBin<AggregateFunctionSumBinKahan>);
}

}
