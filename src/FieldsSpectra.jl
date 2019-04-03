module FieldsSpectra
export compute_shells, compute_shells2D, squared_mean, proj_mean
#export calculate_u1u2_spectrum, calculate_vector_spectrum, calculate_scalar_spectrum
export power_spectrum, power_spectrum!, hpower_spectrum, hpower_spectrum!

using FluidFields, FluidTensors

@inline proj(a::Complex,b::Complex) = muladd(a.re, b.re, a.im*b.im)
@inline proj(u::Vec{<:Complex},v::Vec{<:Complex}) = proj(u.x,v.x) + proj(u.y,v.y) + proj(u.z,v.z)
@inline proj(a::SymTen{<:Complex},b::SymTen{<:Complex}) =
    proj(a.xx,b.xx) + proj(a.yy,b.yy) + proj(a.zz,b.zz) + 2*(proj(a.xy,b.xy) + proj(a.xz,b.xz) + proj(a.yz,b.yz))


function nshells(kx,ky,kz)
    maxdk = max(kx[2],ky[2],kz[2])
    n = round(Int,sqrt(kx[end]^2 + maximum(x->x*x,ky) + maximum(x->x*x,kz))/maxdk) + 1
    return n, maxdk
end

function nshells2D(kx,ky)
    maxdk = max(kx[2],ky[2])
    n = round(Int,sqrt(kx[end]^2 + maximum(x->x*x,ky))/maxdk) + 1
    return n, maxdk
end

function compute_shells(kx::AbstractVector{T},ky::AbstractVector,kz::AbstractVector) where {T}
    Nx = length(kx)
    Ny = length(ky)
    Nz = length(kz)
    nShells, maxdk = nshells(kx,ky,kz)
    kh = zeros(T,nShells)
    numPtsInShell = zeros(Int,nShells)

    @inbounds for k in 1:Nz
        kz2 = kz[k]^2
        for j=1:Ny
            kzy2 = kz2 + ky[j]^2
            for i=1:Nx
                K = sqrt(muladd(kx[i],kx[i],kzy2))
                ii = round(Int,K/maxdk)+1
                kh[ii] += K
                numPtsInShell[ii] += 1
            end
        end
    end
  
    @inbounds @simd for i in 1:length(kh)
        kh[i] = kh[i]/numPtsInShell[i]
    end

    return kh
end

compute_shells(f::AbstractField) = compute_shells(f.kx,f.ky,f.kz)

function compute_shells2D(kx::AbstractVector{T},ky) where {T}
    Nx = length(kx)
    Ny = length(ky)
    nShells2D, maxdk2D = nshells2D(kx,ky)
    kh = zeros(T,nShells2D)
    numPtsInShell2D = zeros(Int,nShells2D)

    @inbounds for j=1:Ny
        ky2 = ky[j]^2
        for i=1:Nx
            K = sqrt(muladd(kx[i],kx[i], ky2))
            ii = round(Int,K/maxdk2D)+1
            kh[ii] += K
            numPtsInShell2D[ii] += 1
        end
    end
  
    @inbounds @simd for i in 1:length(kh)
        kh[i] = kh[i]/numPtsInShell2D[i]
    end

    return kh
end

compute_shells2D(f::AbstractField) = compute_shells2D(f.kx,f.ky)

#function calculate_u1u2_spectrum!(Ef,u::VectorField{T},cplane::Int=1) where {T}
#    isrealspace(u) && fourier!(u)
#    ux = u.c.x
#    uy = u.c.y
#    KX = u.kx
#    KY = u.ky
#    NX = size(u,1)
#    NY = size(u,2)
#    # Initialize the shells to zeros
#    fill!(Ef,zero(T))
#    maxdk2d = max(KX[2],KY[2])
#    nshells = min(NX,NY÷2)
#    @inbounds for j in 1:NY
#        KY2 = KY[j]^2
#        n = round(Int, sqrt(KY2)/maxdk2d) + 1
#        n > nshells && break
#        magsq = abs2(ux[1,j,cplane]) + abs2(uy[1,j,cplane])
#        ee = magsq
#        Ef[n]+=ee
#        for i in 2:NX
#            k = sqrt(muladd(KX[i],KX[i], KY2))
#            n = round(Int, k/maxdk2d) + 1
#            n > nshells && break
#            magsq = abs2(ux[i,j,cplane]) + abs2(uy[i,j,cplane])
#            ee = 2*magsq
#            Ef[n]+=ee
#        end
#    end
#    return Ef
#end
#
#function calculate_u1u2_spectrum(u::VectorField{T},p::Int=1) where {T}
#    NX = size(u,1)
#    NY = size(u,2)
#    Ef = zeros(T,min(NX,NY÷2))
#    return calculate_u1u2_spectrum!(Ef,u,p) 
#end
#
#function calculate_vector_spectrum!(Ef,u::VectorField{T}) where {T}
#    isrealspace(u) && fourier!(u)
#    ux = u.c.x
#    uy = u.c.y
#    uz = u.c.z
#    KX = u.kx
#    KY = u.ky
#    KZ = u.kz
#    NX = size(u,1)
#    NY = size(u,2)
#    NZ = size(u,3)
#    # Initialize the shells to zeros
#    fill!(Ef,zero(T))
#    maxdk2d = max(KX[2],KY[2],KZ[2])
#    vK = KX[2]*KY[2]*KZ[2]
#    nshells = min(NX,NY÷2,NZ÷2)
#
#    fix = oneunit(T)
#    @inbounds for l in 1:NZ
#        KZ2 = KZ[l]^2
#        (round(Int, sqrt(KZ2)/maxdk2d) + 1) > nshells && break
#        @inbounds for j in 1:NY
#            KY2 = KY[j]^2 + KZ2
#            K = sqrt(KY2)
#            n = round(Int, K/maxdk2d) + 1
#            n > nshells && break
#            magsq = abs2(ux[1,j,l]) + abs2(uy[1,j,l]) + abs2(uz[1,j,l])
#            ee = magsq
#            Ef[n]+=ee
#            fix = zero(T)
#            for i in 2:NX
#                K = sqrt(muladd(KX[i],KX[i], KY2))
#                n = round(Int, K/maxdk2d) + 1
#                n > nshells && break
#                magsq = abs2(ux[i,j,l]) + abs2(uy[i,j,l]) + abs2(uz[i,j,l])
#                ee = 2*magsq
#                Ef[n]+=ee
#            end
#        end
#    end
#    return Ef
#end
#
#function calculate_vector_spectrum(u::VectorField{T}) where {T}
#    NX = size(u,1)
#    NY = size(u,2)
#    NZ = size(u,3)
#    Ef = zeros(T,min(NX,NY÷2,NZ÷2))
#    return calculate_vector_spectrum!(Ef,u) 
#end
#
#function calculate_scalar_spectrum!(Ef,u::ScalarField{T}) where {T}
#    isrealspace(u) && fourier!(u)
#    KX = u.kx
#    KY = u.ky
#    KZ = u.kz
#    NX = size(u,1)
#    NY = size(u,2)
#    NZ = size(u,3)
#    # Initialize the shells to zeros
#    fill!(Ef,zero(T))
#    maxdk2d = max(KX[2],KY[2],KZ[2])
#    nshells = min(NX,NY÷2,NZ÷2)
#
#    @inbounds for l in 1:NZ
#        KZ2 = KZ[l]^2
#        (round(Int, sqrt(KZ2)/maxdk2d) + 1) > nshells && break
#        @inbounds for j in 1:NY
#            KY2 = KY[j]^2 + KZ2
#            n = round(Int, sqrt(KY2)/maxdk2d) + 1
#            n > nshells && break
#            magsq = abs2(u[1,j,l])
#            ee = magsq
#            Ef[n]+=ee
#            for i in 2:NX
#                k = sqrt(muladd(KX[i],KX[i], KY2))
#                n = round(Int, k/maxdk2d) + 1
#                n > nshells && break
#                magsq = abs2(u[i,j,l])
#                ee = 2*magsq
#                Ef[n]+=ee
#            end
#        end
#    end
#    return Ef
#end
#
#function calculate_scalar_spectrum(u::ScalarField{T}) where {T}
#    NX = size(u,1)
#    NY = size(u,2)
#    NZ = size(u,3)
#    Ef = zeros(T,min(NX,NY÷2,NZ÷2))
#    return calculate_scalar_spectrum!(Ef,u) 
#end

function squared_mean(u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    ee = zero(T)
    @inbounds for l in axes(u,3)
        ep = zero(T)
        @inbounds for j in axes(u,2)
            ex = zero(T)
            @simd for i in axes(u,1)
                magsq = abs2(u[i,j,l])
                ex += (1 + (i>1))*magsq 
            end
            ep += ex
        end
        ee += ep
    end
    return ee
end

function proj_mean(u::AbstractField{T},v::AbstractField{T2}) where {T,T2}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    ee = zero(promote_type(T,T2))
    @inbounds for l in axes(u,3)
        ep = zero(T)
        @inbounds for j in axes(u,2)
            ex = zero(T)
            @simd for i in axes(u,1)
                magsq = proj(u[i,j,l],v[i,j,l])
                ex += (1 + (i>1))*magsq 
            end
            ep += ex
        end
        ee += ep
    end
    return ee
end

function power_spectrum!(Ef::Vector{T},u::AbstractField,v::AbstractField=u) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    KX = u.kx
    KY = u.ky
    KZ = u.kz
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)
    # Initialize the shells to zeros
    fill!(Ef,zero(T))
    maxdk2d = max(KX[2],KY[2],KZ[2])
    #nshells = min(NX,NY÷2,NZ÷2)
    nshells = length(Ef)

    @inbounds for l in 1:NZ
        KZ2 = KZ[l]^2
        @inbounds for j in 1:NY
            KY2 = KY[j]^2 + KZ2
            n = round(Int, sqrt(KY2)/maxdk2d) + 1
            magsq = proj(u[1,j,l],v[1,j,l])
            ee = magsq
            Ef[n]+=ee
            for i in 2:NX
                k = sqrt(muladd(KX[i],KX[i], KY2))
                n = round(Int, k/maxdk2d) + 1
                magsq = proj(u[i,j,l],v[i,j,l])
                ee = 2*magsq
                Ef[n]+=ee
            end
        end
    end
    return Ef
end

function power_spectrum(u::AbstractField{T},v::AbstractField=u) where {T}
    n,_ = nshells(u.kx,u.ky,u.kz)
    Ef = zeros(T,n)
    return power_spectrum!(Ef,u,v) 
end

function hpower_spectrum!(Ef::Vector{T},u::AbstractField,v::AbstractField=u,p::Integer=1) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    KX = u.kx
    KY = u.ky
    NX = size(u,1)
    NY = size(u,2)
    # Initialize the shells to zeros
    fill!(Ef,zero(T))
    maxdk2d = max(KX[2],KY[2])
    nshells = length(Ef)

    @inbounds for j in 1:NY
        KY2 = KY[j]^2
        n = round(Int, sqrt(KY2)/maxdk2d) + 1
        magsq = proj(u[1,j,p],v[1,j,p])
        ee = magsq
        Ef[n]+=ee
        for i in 2:NX
            k = sqrt(muladd(KX[i],KX[i], KY2))
            n = round(Int, k/maxdk2d) + 1
            magsq = proj(u[i,j,p],v[i,j,p])
            ee = 2*magsq
            Ef[n]+=ee
        end
    end
    return Ef
end

hpower_spectrum!(Ef::Vector{T},u::AbstractField,p::Integer) where {T} = hpower_spectrum!(Ef,u,u,p)

function hpower_spectrum(u::AbstractField{T},v::AbstractField=u,p::Integer=1) where {T}
    n,_ = nshells2D(u.kx,u.ky)
    Ef = zeros(T,n)
    return hpower_spectrum!(Ef,u,v,p) 
end

hpower_spectrum(u::AbstractField,p::Integer) = hpower_spectrum(u,u,p)

end # module
