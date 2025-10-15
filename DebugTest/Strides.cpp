#include <iostream>
#include <stdexcept>
#include <vector>
#include <sstream>

#include <aether/core/config.hpp>
#include <aether/core/strides.hpp>
#include <aether/core/views.hpp>

using namespace aether::core;

static void require(bool cond, const char* msg) {
  if (!cond) throw std::runtime_error(msg);
}

static std::string dims_str(const Extents& e, int ng) {
  std::ostringstream os;
// #if AETHER_DIM == 1
  // os << "DIM=1 nx="<<e.nx<<" Nx="<<e.Nx<<" ng="<<ng;
// #elif AETHER_DIM == 2
  // os << "DIM=2 nx="<<e.nx<<" ny="<<e.ny<<" Nx="<<e.Nx<<" Ny="<<e.Ny<<" ng="<<ng;
// #else
  os << "DIM="<<AETHER_DIM <<" nx="<<e.nx<<" ny="<<e.ny<<" nz="<<e.nz
     <<" Nx="<<e.Nx<<" Ny="<<e.Ny<<" Nz="<<e.Nz<<" ng="<<ng;
// #endif
  return os.str();
}

// ----------------------------- Extents checks ------------------------------
static void check_extents_basic(const Extents& e, int ng) {
  std::cout << "  Extents: " << dims_str(e, ng) << "\n";
  // stride relationships
  require(e.sx == 1, "sx must be 1");
  require(e.sy == (std::size_t)e.Nx, "sy must equal Nx");
  require(e.sz == (std::size_t)e.Nx * (std::size_t)e.Ny, "sz must equal Nx*Ny");

  // index monotonicity / last index
#if AETHER_DIM == 1
  require(e.index(-ng) == 0, "1D: first flat index must be 0");
  require(e.index(e.nx+ng-1) == (std::size_t)e.Nx-1, "1D: last flat index incorrect");
#elif AETHER_DIM == 2
  require(e.index(-ng,-ng) == 0, "2D: first flat index must be 0");
  require(e.index(e.nx+ng-1, e.ny+ng-1) == (std::size_t)e.Nx*e.Ny - 1, "2D: last flat index incorrect");
  // unit steps
  auto a = e.index(0,0);
  require(e.index(1,0) == a + 1, "2D: step i should be +1");
  require(e.index(0,1) == a + e.sy, "2D: step j should be +sy");
#else
  require(e.index(-ng,-ng,-ng) == 0, "3D: first flat index must be 0");
  require(e.index(e.nx+ng-1, e.ny+ng-1, e.nz+ng-1) == (std::size_t)e.Nx*e.Ny*e.Nz - 1, "3D: last flat index incorrect");
  // unit steps
  auto a = e.index(0,0,0);
  require(e.index(1,0,0) == a + 1, "3D: step i should be +1");
  require(e.index(0,1,0) == a + e.sy, "3D: step j should be +sy");
  require(e.index(0,0,1) == a + e.sz, "3D: step k should be +sz");
#endif
}

// ----------------------------- FaceGrid shape/stride checks -----------------
static void check_facegrid_X(const Extents& e) {
  FaceGridX gx(e);
  const std::size_t nfaces = gx.nfaces();
  const std::size_t expect = (std::size_t)(e.Nx + 1) * e.Ny * e.Nz;
  require(nfaces == expect, "FaceGridX: nfaces mismatch (should be (Nx+1)*Ny*Nz)");

  // stride steps: along iF (+1), along j (+sy=NxF), along k (+sz=NxF*Ny)
  auto a = gx.index(0,0,0);
  require(gx.index(1,0,0) == a + 1, "FaceGridX: step iF must be +1");
  if (AETHER_DIM > 1) require(gx.index(0,1,0) == a + gx.sy, "FaceGridX: step j must be +sy");
  if (AETHER_DIM > 2) require(gx.index(0,0,1) == a + gx.sz, "FaceGridX: step k must be +sz");

  // last element is total-1
  require(gx.index(gx.NxF-1, gx.Ny-1, gx.Nz-1) == nfaces - 1, "FaceGridX: last flat index incorrect");

  // bijection test: visit all faces and ensure we see exactly [0..nfaces-1]
  std::vector<char> seen(nfaces, 0);
  for (int k=0; k<gx.Nz; ++k)
    for (int j=0; j<gx.Ny; ++j)
      for (int iF=0; iF<gx.NxF; ++iF) {
        auto f = gx.index(iF,j,k);
        require(f < nfaces, "FaceGridX: flat index out of bounds");
        seen[f] = 1;
      }
  for (std::size_t t=0; t<nfaces; ++t) require(seen[t], "FaceGridX: missing face index");
}

static void check_facegrid_Y(const Extents& e) {
  FaceGridY gy(e);
  const std::size_t nfaces = gy.nfaces();
  const std::size_t expect = (std::size_t)e.Nx * (e.Ny + 1) * e.Nz;
  require(nfaces == expect, "FaceGridY: nfaces mismatch (should be Nx*(Ny+1)*Nz)");

  auto a = gy.index(0,0,0);
  require(gy.index(1,0,0) == a + 1, "FaceGridY: step i must be +sy (=Nx)");
  require(gy.index(0,1,0) == a + gy.sy,      "FaceGridY: step jF must be +1");
  if (AETHER_DIM > 2) require(gy.index(0,0,1) == a + gy.sz, "FaceGridY: step k must be +sz");

  require(gy.index(gy.Nx-1, gy.NyF-1, gy.Nz-1) == nfaces - 1, "FaceGridY: last flat index incorrect");

  std::vector<char> seen(nfaces, 0);
  for (int k=0; k<gy.Nz; ++k)
    for (int jF=0; jF<gy.NyF; ++jF)
      for (int i=0; i<gy.Nx; ++i) {
        auto f = gy.index(i,jF,k);
        require(f < nfaces, "FaceGridY: flat index out of bounds");
        seen[f] = 1;
      }
  for (std::size_t t=0; t<nfaces; ++t) require(seen[t], "FaceGridY: missing face index");
}

static void check_facegrid_Z(const Extents& e) {
  FaceGridZ gz(e);
  const std::size_t nfaces = gz.nfaces();
  const std::size_t expect = (std::size_t)e.Nx * e.Ny * (e.Nz + 1);
  require(nfaces == expect, "FaceGridZ: nfaces mismatch (should be Nx*Ny*(Nz+1))");

  auto a = gz.index(0,0,0);
  require(gz.index(1,0,0) == a + 1, "FaceGridZ: step i must be +sy (=Nx)");
  require(gz.index(0,1,0) == a + gz.sy,      "FaceGridZ: step j must be +1");
  require(gz.index(0,0,1) == a + gz.sz, "FaceGridZ: step kF must be +sz (=Nx*Ny)");

  require(gz.index(gz.Nx-1, gz.Ny-1, gz.NzF-1) == nfaces - 1, "FaceGridZ: last flat index incorrect");

  std::vector<char> seen(nfaces, 0);
  for (int kF=0; kF<gz.NzF; ++kF)
    for (int j=0; j<gz.Ny; ++j)
      for (int i=0; i<gz.Nx; ++i) {
        auto f = gz.index(i,j,kF);
        require(f < nfaces, "FaceGridZ: flat index out of bounds");
        seen[f] = 1;
      }
  for (std::size_t t=0; t<nfaces; ++t) require(seen[t], "FaceGridZ: missing face index");
}

// ----------------------------- FaceArray checks -----------------------------
template<int NCOMP>
static void check_face_arrays(const Extents& e) {
  // X
  {
    FaceGridX gx(e);
    for (int Q : {1,2}) {
      FaceArraySoAT<NCOMP> Fx(gx, Quadrature{Q});
      auto Fv = Fx.view();
      require(Fx.nfaces == gx.nfaces(), "FaceArrayX: nfaces mismatch");
      require(Fx.comp[0].size() == Fx.nfaces * (std::size_t)Q, "FaceArrayX: storage size mismatch");
      // write/read sentinel values
      int iF = gx.NxF-1, j = gx.Ny-1, k = gx.Nz-1;
      int f = face_index(gx, iF, j, k);
      for (int c=0;c<NCOMP;++c)
        for (int q=0;q<Q;++q) {
          double v = c*1e6 + f*10 + q;
          Fv.var(c, f, q) = v;
          require(Fv.var(c, f, q) == v, "FaceArrayX: write/read mismatch");
        }
    }
  }
  // Y
  if (AETHER_DIM > 1)
  {
    FaceGridY gy(e);
    for (int Q : {1,2}) {
      FaceArraySoAT<NCOMP> Fy(gy, Quadrature{Q});
      auto Fv = Fy.view();
      require(Fy.nfaces == gy.nfaces(), "FaceArrayY: nfaces mismatch");
      require(Fy.comp[0].size() == Fy.nfaces * (std::size_t)Q, "FaceArrayY: storage size mismatch");
      int i = gy.Nx-1, jF = gy.NyF-1, k = gy.Nz-1;
      int f = face_index(gy, i, jF, k);
      for (int c=0;c<NCOMP;++c)
        for (int q=0;q<Q;++q) {
          double v = c*1e6 + f*10 + q;
          Fv.var(c, f, q) = v;
          require(Fv.var(c, f, q) == v, "FaceArrayY: write/read mismatch");
        }
    }
  }
  // Z
  if (AETHER_DIM > 2)
  {
    FaceGridZ gz(e);
    for (int Q : {1,2}) {
      FaceArraySoAT<NCOMP> Fz(gz, Quadrature{Q});
      auto Fv = Fz.view();
      require(Fz.nfaces == gz.nfaces(), "FaceArrayZ: nfaces mismatch");
      require(Fz.comp[0].size() == Fz.nfaces * (std::size_t)Q, "FaceArrayZ: storage size mismatch");
      int i = gz.Nx-1, j = gz.Ny-1, kF = gz.NzF-1;
      int f = face_index(gz, i, j, kF);
      for (int c=0;c<NCOMP;++c)
        for (int q=0;q<Q;++q) {
          double v = c*1e6 + f*10 + q;
          Fv.var(c, f, q) = v;
          require(Fv.var(c, f, q) == v, "FaceArrayZ: write/read mismatch");
        }
    }
  }
}

// ----------------------------- Cells checks ---------------------------------
template<int NCOMP>
static void check_cells(const Extents& e, int ng) {
  std::cout << "  Cells<NCOMP="<<NCOMP<<">: " << dims_str(e, ng) << "\n";
  CellsSoAT<NCOMP> U(
#if AETHER_DIM == 1
    e.nx, 0, 0, ng
#elif AETHER_DIM == 2
    e.nx, e.ny, 0, ng
#else
    e.nx, e.ny, e.nz, ng
#endif
  );
  auto V = U.view();
  require(U.size_flat() == (std::size_t)e.Nx*e.Ny*e.Nz, "Cells: flat size mismatch");

  // simple pattern via (i,j,k)
#if AETHER_DIM == 1
  for (int c=0;c<NCOMP;++c)
    for (int i=-ng;i<e.nx+ng;++i)
      V.var(c,i) = (double)(c*1e6 + V.idx(i));
#elif AETHER_DIM == 2
  for (int c=0;c<NCOMP;++c)
    for (int j=-ng;j<e.ny+ng;++j)
      for (int i=-ng;i<e.nx+ng;++i)
        V.var(c,i,j) = (double)(c*1e6 + V.idx(i,j));
#else
  for (int c=0;c<NCOMP;++c)
    for (int k=-ng;k<e.nz+ng;++k)
      for (int j=-ng;j<e.ny+ng;++j)
        for (int i=-ng;i<e.nx+ng;++i)
          V.var(c,i,j,k) = (double)(c*1e6 + V.idx(i,j,k));
#endif

  // verify a few corners and midpoints match flat addressing
  auto check = [&](int i, int j, int k){
    auto flat = e.index(i,j,k);
    for (int c=0;c<NCOMP;++c) {
      double v3 = V.var(c,i,j,k);
      double vf = V.var(c, flat);
      require(v3 == vf, "Cells: 3D vs flat accessor mismatch");
    }
  };
#if AETHER_DIM == 1
  check(-ng,0,0); check(e.nx+ng-1,0,0);
#elif AETHER_DIM == 2
  check(-ng,-ng,0); check(e.nx+ng-1, e.ny+ng-1, 0);
  if (e.nx>2 && e.ny>2) check(0,0,0);
#else
  check(-ng,-ng,-ng); check(e.nx+ng-1, e.ny+ng-1, e.nz+ng-1);
  if (e.nx>2 && e.ny>2 && e.nz>2) check(0,0,0);
#endif
}

// ----------------------------- Main -----------------------------------------
int main() {
  try {
    std::cout << "AETHER grid/face smoke++\n";
#if AETHER_DIM == 1
    const int nx=7, ng=2;
    Extents e(nx, ng);                 // your tolerant ctor for 1D
#elif AETHER_DIM == 2
    const int nx=5, ny=3, ng=1;
    Extents e(nx, ny, ng);             // tolerant (nx, ny, ng)
#else
    const int nx=4, ny=3, nz=2, ng=1;
    Extents e(nx, ny, nz, ng);         // 3D canonical
#endif

    check_extents_basic(e, 
#if AETHER_DIM == 1
      2
#elif AETHER_DIM == 2
      1
#else
      1
#endif
    );

    // Cells (use Euler-like component count=5 for structure testing)
    check_cells<5>(e,
#if AETHER_DIM == 1
      2
#else
      1
#endif
    );

    // Face grids
    check_facegrid_X(e);
    if (AETHER_DIM > 1) check_facegrid_Y(e);
    if (AETHER_DIM > 2) check_facegrid_Z(e);

    // Face arrays with Q=1,2
    check_face_arrays<5>(e);

    std::cout << "All structural checks passed.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << "\n";
    return 1;
  }
}
