module FieldsSpectra
export compute_shells, compute_shells2D, calculate_u1u2_spectrum, calculate_u_spectrum

using FluidFields

function compute_shells(kx::AbstractVector,ky::AbstractVector,kz::AbstractVector)
    Nx = length(kx)
    Ny = length(ky)
    Nz = length(kz)
    nShells = min(Nx,Ny÷2,Nz÷2)
    maxdk = max(kx[2],ky[2],kz[2])
    kh = zeros(nShells)
    numPtsInShell = zeros(Int,nShells)

    @inbounds for k in 1:Nz
        kz2 = kz[k]^2
        (round(Int,sqrt(kz2)/maxdk)+1) > nShells && break
        for j=1:Ny
            kzy2 = kz2 + ky[j]^2
            (round(Int,sqrt(kzy2)/maxdk)+1) > nShells && break
            for i=1:Nx
                K = sqrt(muladd(kx[i],kx[i],kzy2))
                ii = round(Int,K/maxdk)+1
                ii > nShells && break
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

function compute_shells2D(kx,ky)
    Nx = length(kx)
    Ny = length(ky)
    nShells2D = min(Nx,Ny÷2)
    maxdk2D = max(kx[2],ky[2])
    kh = zeros(nShells2D)
    numPtsInShell2D = zeros(Int,nShells2D)

    @inbounds for j=1:Ny
        ky2 = ky[j]^2
        (round(Int,sqrt(ky2)/maxdk2D)+1) > nShells2D && break
        for i=1:Nx
            K = sqrt(muladd(kx[i],kx[i], ky2))
            ii = round(Int,K/maxdk2D)+1
            ii > nShells2D && break
            kh[ii] += K
            numPtsInShell2D[ii] += 1
        end
    end
  
    @inbounds @simd for i in 1:length(kh)
        kh[i] = kh[i]/numPtsInShell2D[i]
    end

    return kh
end

function calculate_u1u2_spectrum!(Ef,u::VectorField,cplane::Int=1)
    isrealspace(u) && fourier!(u)
    ux = u.c.x
    uy = u.c.y
    KX = u.kx
    KY = u.ky
    NX = size(u,1)
    NY = size(u,2)
    # Initialize the shells to zeros
    fill!(Ef,0)
    maxdk2d = max(KX[2],KY[2])
    nshells = min(NX,NY÷2)
    @inbounds for j in 1:NY
        KY2 = KY[j]^2
        n = round(Int, sqrt(KY2)/maxdk2d) + 1
        n > nshells && break
        magsq = abs2(ux[1,j,cplane]) + abs2(uy[1,j,cplane])
        ee = 0.5 * magsq / maxdk2d
        Ef[n]+=ee
        for i in 2:NX
            k = sqrt(muladd(KX[i],KX[i], KY2))
            n = round(Int, k/maxdk2d) + 1
            n > nshells && break
            magsq = abs2(ux[i,j,cplane]) + abs2(uy[i,j,cplane])
            ee = magsq / maxdk2d
            Ef[n]+=ee
        end
    end
    return Ef
end

function calculate_u1u2_spectrum(u::VectorField,p::Int=1)
    NX = size(u,1)
    NY = size(u,2)
    Ef = zeros(Float64,min(NX,NY÷2))
    return calculate_u1u2_spectrum!(Ef,u,p) 
end

function calculate_u_spectrum!(Ef,u::VectorField)
    isrealspace(u) && fourier!(u)
    ux = u.c.x
    uy = u.c.y
    uz = u.c.z
    KX = u.kx
    KY = u.ky
    KZ = u.kz
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)
    # Initialize the shells to zeros
    fill!(Ef,0)
    maxdk2d = max(KX[2],KY[2],KZ[2])
    nshells = min(NX,NY÷2,NZ÷2)

    @inbounds for l in 1:NZ
        KZ2 = KZ[l]^2
        (round(Int, sqrt(KZ2)/maxdk2d) + 1) > nshells && break
        @inbounds for j in 1:NY
            KY2 = KY[j]^2 + KZ2
            n = round(Int, sqrt(KY2)/maxdk2d) + 1
            n > nshells && break
            magsq = abs2(ux[1,j,l]) + abs2(uy[1,j,l]) + abs2(uz[1,j,l])
            ee = 0.5 * magsq / maxdk2d
            Ef[n]+=ee
            for i in 2:NX
                k = sqrt(muladd(KX[i],KX[i], KY2))
                n = round(Int, k/maxdk2d) + 1
                n > nshells && break
                magsq = abs2(ux[i,j,l]) + abs2(uy[i,j,l]) + abs2(uz[i,j,l])
                ee = magsq / maxdk2d
                Ef[n]+=ee
            end
        end
    end
    return Ef
end

function calculate_u_spectrum(u::VectorField)
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)
    Ef = zeros(Float64,min(NX,NY÷2,NZ÷2))
    return calculate_u_spectrum!(Ef,u) 
end

end # module
