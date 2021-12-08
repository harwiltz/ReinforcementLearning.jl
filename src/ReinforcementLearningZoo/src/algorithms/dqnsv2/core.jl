abstract type AbstractDQNLearner <: AbstractLearner end

abstract type AbstractQNetwork end

export AbstractQNetwork, BasicQNetwork, DoubleBasicQNetwork, @with_2q, q_values

macro with_2q(ex::Expr)
    local with_2q__type = ex.args[2]
    local with_2q__paramtypename = string(with_2q__type.args[1])
    local with_2q__name_params = split(with_2q__paramtypename, '{')
    local with_2q__typename = with_2q__name_params[1]
    local with_2q__typeparams = Meta.parse("{" * with_2q__name_params[2])
    local with_2q__doubletype = Meta.parse("Double" * with_2q__typename)

    local with_2q__fields = ex.args[3].args

    return esc(quote
        $ex

        mutable struct $(Meta.parse("Double" * string(with_2q__type)))
            $(with_2q__fields...)
            target_approximator::Any
        end

        function q_values(learner::$with_2q__doubletype, state)
            "this is from Double"
        end
    end)
end

@with_2q mutable struct BasicQNetwork{Tq} <: AbstractQNetwork
    approximator::Tq
end

function q_values(learner::AbstractQNetwork, state)
    "yo yo ma"
end
