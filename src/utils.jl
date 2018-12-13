

if !isdefined(@__MODULE__, :__EXPRESSION_HASHES__)
    __EXPRESSION_HASHES__ = Set{AbstractString}()
end

"""
If loaded twice without changes, evaluate expression only for the first time.
This is useful for reloading code in REPL. For example, the following code will
produce `invalid redifinition` error if loaded twice:

    type Point{T}
        x::T
        y::T
    end

Wrapped into @runonce, however, the code is reloaded fine:

    @runonce type Point{T}
        x::T
        y::T
    end

@runonce doesn't have any affect on expression itself.
"""
macro runonce(expr)
    h = string(expr)
    return esc(quote
        if !in($h, __EXPRESSION_HASHES__)
            push!(__EXPRESSION_HASHES__, $h)
            $expr
        end
    end)
end


"""Same as `get` function, but evaluates default_expr only if needed"""
macro get(dict, key, default_expr)
    return esc(quote
        if haskey($dict, $key)
            $dict[$key]
        else
            $default_expr
        end
    end)
end


"""
Same as `@get`, but creates new object from `default_expr` if
it didn't exist before
"""
macro get_or_create(dict, key, default_expr)
    return esc(quote
        if !haskey($dict, $key)
            $dict[$key] = $default_expr
        end
        $dict[$key]
    end)
end



"""
Same as `@get`, but immediately exits function and return `default_expr`
if key doesn't exist.
"""
macro get_or_return(dict, key, default_expr)
    return esc(quote
        if haskey($dict, $key)
            $dict[$key]
        else
            return $default_expr
            nothing  # not reachable, but without it code won't compile
        end
    end)
end

"""
Get array of size `sz` from a `dict` by `key`. If element doesn't exist or
its size is not equal to `sz`, create and return new array
using `default_expr`. If element exists, but is not an error,
throw ArgumentError.
"""
macro get_array(dict, key, sz, default_expr)
    return esc(quote
        if (haskey($dict, $key) && !isa($dict[$key], Array))
           let
              local k = $key
              throw(ArgumentError("Key `$k` exists, but is not an array"))
           end
        end
        if (!haskey($dict, $key) || size($dict[$key]) != $sz)
            # ensure $default_expr results in an ordinary array
            $dict[$key] = convert(Array, $default_expr)
        end
        $dict[$key]
    end)
end

# maybe these could be moved to rbm.jl ##
function logistic(x)
    return 1 ./ (1 .+ exp.(-x))
end

function IsingActivation(x)
  return 1 ./ (1 .+ exp.(-2x))
end

##########################################

const KNOWN_OPTIONS =
    [:debug, :gradient, :update, :sampler, :scorer, :reporter,
     :batch_size, :n_epochs, :n_gibbs,
     :lr, :momentum, :weight_decay_kind, :weight_decay_rate,
     :sparsity_cost, :sparsity_target,
     :randomize,
     # deprecated options
     :n_iter]
const DEPRECATED_OPTIONS = Dict(:n_iter => :n_epochs)

function check_options(opts::Dict)
    deprecated_keys = keys(DEPRECATED_OPTIONS)
    debug = get(opts, :debug, true)

    for opt in keys(opts)
        if debug && !in(opt, KNOWN_OPTIONS)
            @warn("Option '$opt' is unknown, ignoring")
        end
        if debug && in(opt, deprecated_keys)
            @warn("Option '$opt' is deprecated, " *
                 "use '$(DEPRECATED_OPTIONS[opt])' instead")
        end
    end
end


function split_evenly(n, len)
    n_parts = Int(ceil(n / len))
    parts = Array{Tuple}(undef, n_parts)
    for i=1:n_parts
        start_idx = (i-1)*len + 1
        end_idx = min(i*len, n)
        parts[i] = (start_idx, end_idx)
    end
    return parts
end

"""
`tofinite!` takes an array and
1. turns all NaNs to zeros
2. turns all Infs and -Infs to the largest and
   smallest representable values accordingly
3. turns all Missings to `miss_val` (0.0 by default)
4. turns all zeros to the smallest representable
   non-zero values, if `nozeros` is true
"""
function tofinite!(x::Array; nozeros=false, miss_val = 0.0)
    for i in eachindex(x)
        if ismissing(x[i])
	    print(x[i])
	    print(typeof(x))
	    x[i] = miss_val
        elseif isnan(x[i])
            x[i] = 0.0
        elseif isinf(x[i])
            if x[i] > 0.0
                x[i] = prevfloat(x[i])
            else
                x[i] = nextfloat(x[i])
            end
        end

        if x[i] == 0.0 && nozeros
            x[i] = nextfloat(x[i])
        end
    end
end


function ensure_type(newT::DataType, A::AbstractArray)
    if eltype(A) != newT
        map(newT, A)
    else
        A
    end
end

function add!(X::Array{T}, inc::T) where T
    @simd for i=1:length(X)
        @inbounds X[i] += inc
    end
end
