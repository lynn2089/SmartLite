#pragma once

#include <experimental/type_traits>
#include <type_traits>

#include <IO/WriteHelpers.h>
#include <IO/ReadHelpers.h>

#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypesDecimal.h>
#include <Columns/ColumnVector.h>

#include <AggregateFunctions/IAggregateFunction.h>
#include <common/extended_types.h>
#include <common/defines.h>


namespace DB
{
struct Settings;

template <typename T>
struct AggregateFunctionSumBinAddOverflowImpl
{
    static void NO_SANITIZE_UNDEFINED ALWAYS_INLINE add(T & lhs, const T & rhs)
    {
        if constexpr (is_big_int_v<T>){
            lhs = lhs << 1;
            lhs |= rhs;
        }else{
            lhs += rhs;
        }
        
    }
};
template <typename DecimalNativeType>
struct AggregateFunctionSumBinAddOverflowImpl<Decimal<DecimalNativeType>>
{
    static void NO_SANITIZE_UNDEFINED ALWAYS_INLINE add(Decimal<DecimalNativeType> & lhs, const Decimal<DecimalNativeType> & rhs)
    {
        lhs.addOverflow(rhs);
    }
};

template <typename T>
struct AggregateFunctionSumBinData
{
    using Impl = AggregateFunctionSumBinAddOverflowImpl<T>;
    T sum{};

    void NO_SANITIZE_UNDEFINED ALWAYS_INLINE add(T value)
    {
        Impl::add(sum, value);
    }

    /// Vectorized version
    template <typename Value>
    void NO_SANITIZE_UNDEFINED NO_INLINE addMany(const Value * __restrict ptr, size_t count)
    {
        const auto * end = ptr + count;

        if constexpr (std::is_floating_point_v<T>)
        {
            constexpr size_t unroll_count = 128 / sizeof(T);
            T partial_sums[unroll_count]{};

            const auto * unrolled_end = ptr + (count / unroll_count * unroll_count);

            while (ptr < unrolled_end)
            {
                for (size_t i = 0; i < unroll_count; ++i)
                    Impl::add(partial_sums[i], ptr[i]);
                ptr += unroll_count;
            }

            for (size_t i = 0; i < unroll_count; ++i)
                Impl::add(sum, partial_sums[i]);
        }

        T local_sum{};
        while (ptr < end)
        {
            Impl::add(local_sum, *ptr);
            ++ptr;
        }
        Impl::add(sum, local_sum);
    }

    template <typename Value>
    void NO_SANITIZE_UNDEFINED NO_INLINE addManyNotNull(const Value * __restrict ptr, const UInt8 * __restrict null_map, size_t count)
    {
        const auto * end = ptr + count;

        if constexpr (std::is_floating_point_v<T>)
        {
            constexpr size_t unroll_count = 128 / sizeof(T);
            T partial_sums[unroll_count]{};

            const auto * unrolled_end = ptr + (count / unroll_count * unroll_count);

            while (ptr < unrolled_end)
            {
                for (size_t i = 0; i < unroll_count; ++i)
                {
                    if (!null_map[i])
                    {
                        Impl::add(partial_sums[i], ptr[i]);
                    }
                }
                ptr += unroll_count;
                null_map += unroll_count;
            }

            for (size_t i = 0; i < unroll_count; ++i)
                Impl::add(sum, partial_sums[i]);
        }

        T local_sum{};
        while (ptr < end)
        {
            if (!*null_map)
                Impl::add(local_sum, *ptr);
            ++ptr;
            ++null_map;
        }
        Impl::add(sum, local_sum);
    }

    void NO_SANITIZE_UNDEFINED merge(const AggregateFunctionSumBinData & rhs)
    {
        Impl::add(sum, rhs.sum);
    }

    void write(WriteBuffer & buf) const
    {
        writeBinary(sum, buf);
    }

    void read(ReadBuffer & buf)
    {
        readBinary(sum, buf);
    }

    T get() const
    {
        return sum;
    }

};

template <typename T>
struct AggregateFunctionSumBinKahanData
{
    static_assert(std::is_floating_point_v<T>,
        "It doesn't make sense to use Kahan Summation algorithm for non floating point types");

    T sum{};
    T compensation{};

    template <typename Value>
    void ALWAYS_INLINE addImpl(Value value, T & out_sum, T & out_compensation)
    {
        auto compensated_value = static_cast<T>(value) - out_compensation;
        auto new_sum = out_sum + compensated_value;
        out_compensation = (new_sum - out_sum) - compensated_value;
        out_sum = new_sum;
    }

    void ALWAYS_INLINE add(T value)
    {
        addImpl(value, sum, compensation);
    }

    /// Vectorized version
    template <typename Value>
    void NO_INLINE addMany(const Value * __restrict ptr, size_t count)
    {
        constexpr size_t unroll_count = 4;
        T partial_sums[unroll_count]{};
        T partial_compensations[unroll_count]{};

        const auto * end = ptr + count;
        const auto * unrolled_end = ptr + (count / unroll_count * unroll_count);

        while (ptr < unrolled_end)
        {
            for (size_t i = 0; i < unroll_count; ++i)
                addImpl(ptr[i], partial_sums[i], partial_compensations[i]);
            ptr += unroll_count;
        }

        for (size_t i = 0; i < unroll_count; ++i)
            mergeImpl(sum, compensation, partial_sums[i], partial_compensations[i]);

        while (ptr < end)
        {
            addImpl(*ptr, sum, compensation);
            ++ptr;
        }
    }

    template <typename Value>
    void NO_INLINE addManyNotNull(const Value * __restrict ptr, const UInt8 * __restrict null_map, size_t count)
    {
        constexpr size_t unroll_count = 4;
        T partial_sums[unroll_count]{};
        T partial_compensations[unroll_count]{};

        const auto * end = ptr + count;
        const auto * unrolled_end = ptr + (count / unroll_count * unroll_count);

        while (ptr < unrolled_end)
        {
            for (size_t i = 0; i < unroll_count; ++i)
                if (!null_map[i])
                    addImpl(ptr[i], partial_sums[i], partial_compensations[i]);
            ptr += unroll_count;
            null_map += unroll_count;
        }

        for (size_t i = 0; i < unroll_count; ++i)
            mergeImpl(sum, compensation, partial_sums[i], partial_compensations[i]);

        while (ptr < end)
        {
            if (!*null_map)
                addImpl(*ptr, sum, compensation);
            ++ptr;
            ++null_map;
        }
    }

    void ALWAYS_INLINE mergeImpl(T & to_sum, T & to_compensation, T from_sum, T from_compensation)
    {
        auto raw_sum = to_sum + from_sum;
        auto rhs_compensated = raw_sum - to_sum;
        auto compensations = ((from_sum - rhs_compensated) + (to_sum - (raw_sum - rhs_compensated))) + compensation + from_compensation;
        to_sum = raw_sum + compensations;
        to_compensation = compensations - (to_sum - raw_sum);
    }

    void merge(const AggregateFunctionSumBinKahanData & rhs)
    {
        mergeImpl(sum, compensation, rhs.sum, rhs.compensation);
    }

    void write(WriteBuffer & buf) const
    {
        writeBinary(sum, buf);
        writeBinary(compensation, buf);
    }

    void read(ReadBuffer & buf)
    {
        readBinary(sum, buf);
        readBinary(compensation, buf);
    }

    T get() const
    {
        return sum;
    }
};


enum AggregateFunctionSumBinType
{
    AggregateFunctionTypeSumBin,
    AggregateFunctionTypeSumBinWithOverflow,
    AggregateFunctionTypeSumBinKahan,
};

template <typename T, typename TResult, typename Data, AggregateFunctionSumBinType Type>
class AggregateFunctionSumBin final : public IAggregateFunctionDataHelper<Data, AggregateFunctionSumBin<T, TResult, Data, Type>>
{
public:
    static constexpr bool DateTime64Supported = false;

    using ResultDataType = std::conditional_t<IsDecimalNumber<T>, DataTypeDecimal<TResult>, DataTypeNumber<TResult>>;
    using ColVecType = std::conditional_t<IsDecimalNumber<T>, ColumnDecimal<T>, ColumnVector<T>>;
    using ColVecResult = std::conditional_t<IsDecimalNumber<T>, ColumnDecimal<TResult>, ColumnVector<TResult>>;

    String getName() const override
    {
        if constexpr (Type == AggregateFunctionTypeSumBin)
            return "sumbin";
        else if constexpr (Type == AggregateFunctionTypeSumBinWithOverflow)
            return "sumbinWithOverflow";
        else if constexpr (Type == AggregateFunctionTypeSumBinKahan)
            return "sumbinKahan";
        __builtin_unreachable();
    }

    AggregateFunctionSumBin(const DataTypes & argument_types_)
        : IAggregateFunctionDataHelper<Data, AggregateFunctionSumBin<T, TResult, Data, Type>>(argument_types_, {})
        , scale(0)
    {}

    AggregateFunctionSumBin(const IDataType & data_type, const DataTypes & argument_types_)
        : IAggregateFunctionDataHelper<Data, AggregateFunctionSumBin<T, TResult, Data, Type>>(argument_types_, {})
        , scale(getDecimalScale(data_type))
    {}

    DataTypePtr getReturnType() const override
    {
        if constexpr (IsDecimalNumber<T>)
            return std::make_shared<ResultDataType>(ResultDataType::maxPrecision(), scale);
        else
            return std::make_shared<ResultDataType>();
    }

    bool allocatesMemoryInArena() const override { return false; }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        const auto & column = assert_cast<const ColVecType &>(*columns[0]);
        if constexpr (is_big_int_v<T>)
            this->data(place).add(static_cast<TResult>(column.getData()[row_num]));
        else
            this->data(place).add(column.getData()[row_num]);
    }

    /// Vectorized version when there is no GROUP BY keys.
    void addBatchSinglePlace(
        size_t batch_size, AggregateDataPtr place, const IColumn ** columns, Arena * arena, ssize_t if_argument_pos) const override
    {
        if (if_argument_pos >= 0)
        {
            const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
            for (size_t i = 0; i < batch_size; ++i)
            {
                if (flags[i])
                    add(place, columns, i, arena);
            }
        }
        else
        {
            const auto & column = assert_cast<const ColVecType &>(*columns[0]);
            this->data(place).addMany(column.getData().data(), batch_size);
        }
    }

    void addBatchSinglePlaceNotNull(
        size_t batch_size, AggregateDataPtr place, const IColumn ** columns, const UInt8 * null_map, Arena * arena, ssize_t if_argument_pos)
        const override
    {
        if (if_argument_pos >= 0)
        {
            const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
            for (size_t i = 0; i < batch_size; ++i)
                if (!null_map[i] && flags[i])
                    add(place, columns, i, arena);
        }
        else
        {
            const auto & column = assert_cast<const ColVecType &>(*columns[0]);
            this->data(place).addManyNotNull(column.getData().data(), null_map, batch_size);
        }
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf) const override
    {
        this->data(place).write(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, Arena *) const override
    {
        this->data(place).read(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        auto & column = assert_cast<ColVecResult &>(to);
        column.getData().push_back(this->data(place).get());
    }

private:
    UInt32 scale;
};

}

