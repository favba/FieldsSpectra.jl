module FieldsSpectra
export compute_shells, compute_shells2D, squared_mean, proj_mean
#export calculate_u1u2_spectrum, calculate_vector_spectrum, calculate_scalar_spectrum
export power_spectrum, power_spectrum!, hpower_spectrum, hpower_spectrum!, spectrum3D, spectrum3D!
export hvproj_mean

using FluidFields, FluidTensors

@inline proj(a::Complex,b::Complex) = muladd(a.re, b.re, a.im*b.im)
@inline proj(u::Vec{<:Complex},v::Vec{<:Complex}) = proj(u.x,v.x) + proj(u.y,v.y) + proj(u.z,v.z)
@inline proj(a::SymTen{<:Complex},b::SymTen{<:Complex}) =
    proj(a.xx,b.xx) + proj(a.yy,b.yy) + proj(a.zz,b.zz) + 2*(proj(a.xy,b.xy) + proj(a.xz,b.xz) + proj(a.yz,b.yz))

@inline vecouterproj(a::Vec{<:Complex},b::Vec{<:Complex}) = Vec(proj(a.x,b.x),
                                                                proj(a.y,b.y),
                                                                proj(a.z,b.z))

@inline vecouterproj(a::Ten{<:Complex},b::Ten{<:Complex}) = Vec(proj(a.xx,b.xx) + proj(a.xy,b.yx) + proj(a.xz,b.zx),
                                                                proj(a.yx,b.xy) + proj(a.yy,b.yy) + proj(a.yz,b.zy),
                                                                proj(a.zx,b.xz) + proj(a.zy,b.yz) + proj(a.zz,b.zz))

@inline vecouterproj(a::SymTen{<:Complex},b::Ten{<:Complex}) = Vec(proj(a.xx,b.xx) + proj(a.xy,b.yx) + proj(a.xz,b.zx),
                                                                   proj(a.xy,b.xy) + proj(a.yy,b.yy) + proj(a.yz,b.zy),
                                                                   proj(a.xz,b.xz) + proj(a.yz,b.yz) + proj(a.zz,b.zz))

@inline vecouterproj(a::Ten{<:Complex},b::SymTen{<:Complex}) = Vec(proj(a.xx,b.xx) + proj(a.xy,b.xy) + proj(a.xz,b.xz),
                                                                   proj(a.yx,b.xy) + proj(a.yy,b.yy) + proj(a.yz,b.yz),
                                                                   proj(a.zx,b.xz) + proj(a.zy,b.yz) + proj(a.zz,b.zz))

@inline vecouterproj(a::SymTen{<:Complex},b::SymTen{<:Complex}) = Vec(proj(a.xx,b.xx) + proj(a.xy,b.xy) + proj(a.xz,b.xz),
                                                                      proj(a.xy,b.xy) + proj(a.yy,b.yy) + proj(a.yz,b.yz),
                                                                      proj(a.xz,b.xz) + proj(a.yz,b.yz) + proj(a.zz,b.zz))

@inline vecouterproj(a::SymTen{<:Complex},b::AntiSymTen{<:Complex}) = Vec(proj(a.xy,-b.xy) + proj(a.xz,-b.xz),
                                                                          proj(a.xy,b.xy) + proj(a.yz,-b.yz),
                                                                          proj(a.xz,b.xz) + proj(a.yz,b.yz))

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

function proj_mean(u::AbstractField{T},v::AbstractField{T2}=u) where {T,T2}
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

function hvproj_mean(u::AbstractField{T},v::AbstractField{T2}=u) where {T,T2}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    eeh = zero(promote_type(T,T2))
    eev = zero(promote_type(T,T2))
    @inbounds for l in axes(u,3)
        eph = zero(T)
        epv = zero(T)
        @inbounds for j in axes(u,2)
            exh = zero(T)
            exv = zero(T)
            @simd for i in axes(u,1)
                magsq = vecouterproj(u[i,j,l],v[i,j,l])
                exv += (1 + (i>1))*magsq.z
                exh += (1 + (i>1))*(magsq.x + magsq.y)
            end
            eph += exh
            epv += exv
        end
        eeh += eph
        eev += epv
    end
    return eeh,eev
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

function spectrum3D!(Ef::AbstractArray{T,3},u::AbstractField,v::AbstractField=u) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)

    @inbounds for l in 1:NZ
        @inbounds for j in 1:NY
            for i in 1:NX
                Ef[i,j,l] = proj(u[i,j,l],v[i,j,l])
            end
        end
    end
    return Ef
end

function spectrum3D(u::AbstractField{T},v::AbstractField=u) where {T}
    Ef = zeros(T,size(u))
    return spectrum3D!(Ef,u,v) 
end

end # module