!----------------------------------------------------------------------
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      REAL(8), PARAMETER :: PI = 4 * atan(1.0)
      DOUBLE PRECISION Th, Thp, S, A, B, X, Y

       A = PAR(1)
       B = PAR(2)
       Th=U(1)
       Thp=U(2)
       S=U(3)
       X=U(4)
       Y=U(5)
       F(1)=Thp
       F(2)=B*S*COS(Th)-A*4*PI*COS(4*PI*S)
       F(3)=1
       F(4)=COS(Th)
       F(5)=SIN(Th)

      END SUBROUTINE FUNC
!----------------------------------------------------------------------

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      REAL(8), PARAMETER :: PI = 4 * atan(1.0)
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

       PAR(1)=0.001
       PAR(2)=0.0
       U(1)=0.5*T*PI
       U(2)=0.0
       U(3)=T
       U(4)=0.0
       U(5)=0.0

      END SUBROUTINE STPNT
!----------------------------------------------------------------------

      SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), NBC, IJAC
      DOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)
      DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
      DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)
      REAL(8), PARAMETER :: PI = 4 * atan(1.0)

       FB(1)=U0(1)
       FB(2)=U1(1)-PI*0.5
       FB(3)=U0(3)
       FB(4)=U0(4)
       FB(5)=U0(5)

      END SUBROUTINE BCND
!----------------------------------------------------------------------
      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      !----------------------------------------------------------------------

      SUBROUTINE PVLS(NDIM,U,PAR)
      !     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)

      DOUBLE PRECISION, EXTERNAL :: GETP,GETU2
      INTEGER NDX,NCOL,NTST
      !----------------------------------------------------------------------
      ! NOTE :
      ! Parameters set in this subroutine should be considered as ``solution
      ! measures'' and be used for output purposes only.
      !
      ! They should never be used as `true'' continuation parameters.
      !
      ! They may, however, be added as ``over-specified parameters'' in the
      ! parameter list associated with the AUTO-Constant NICP, in order to
      ! print their values on the screen and in the ``p.xxx file.
      !
      ! They may also appear in the list associated with AUTO-constant NUZR.
      !
      !----------------------------------------------------------------------
      ! For algebraic problems the argument U is, as usual, the state vector.
      ! For differential equations the argument U represents the approximate
      ! solution on the entire interval [0,1]. In this case its values can
      ! be accessed indirectly by calls to GETP, as illustrated below, or
      ! by obtaining NDIM, NCOL, NTST via GETP and then dimensioning U as
      ! U(NDIM,0:NCOL*NTST) in a seperate subroutine that is called by PVLS.
      !----------------------------------------------------------------------

      ! Set PAR(4) equal to the value of U(2) at the left boundary.
       PAR(3)=GETP('BV1',1,U)
       PAR(4)=GETP('BV1',4,U)
       PAR(5)=GETP('BV1',5,U)

      !----------------------------------------------------------------------
      ! The first argument of GETP may be one of the following:
      !        'NRM' (L2-norm),     'MAX' (maximum),
      !        'INT' (integral),    'BV0 (left boundary value),
      !        'MIN' (minimum),     'BV1' (right boundary value).
      !        'MNT' (t value for minimum)
      !        'MXT' (t value for maximum)
      !        'NDIM', 'NDX' (effective (active) number of dimensions)
      !        'NTST' (NTST from constant file)
      !        'NCOL' (NCOL from constant file)
      !        'NBC'  (active NBC)
      !        'NINT' (active NINT)
      !        'DTM'  (delta t for all t values, I=1...NTST)
      !        'WINT' (integration weights used for interpolation, I=0...NCOL)
      !
      ! Also available are
      !   'STP' (Pseudo-arclength step size used).
      !   'FLD' (`Fold function', which vanishes at folds).
      !   'BIF' (`Bifurcation function', which vanishes at singular points).
      !   'HBF' (`Hopf function'; which vanishes at Hopf points).
      !   'SPB' ( Function which vanishes at secondary periodic bifurcations).
      !   'EIG' ( Eigenvalues/multipliers, I=1...2*NDIM, alternates real/imag parts).
      !   'STA' ( Number of stable eigenvalues/multipliers).
      !----------------------------------------------------------------------

      END SUBROUTINE PVLS

!----------------------------------------------------------------------
