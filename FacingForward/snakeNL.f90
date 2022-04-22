!----------------------------------------------------------------------
!----------------------------------------------------------------------
! AUTO forward-looking snake
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
      DOUBLE PRECISION Th, Thp, S, l, ba, X, Y

       l = PAR(1)
       ba = PAR(2)

       Th = U(1)
       Thp = U(2)

       X = U(3)
       Y = U(4)
       S = U(5)

       F(1) = Thp
       F(2) = l**3*S*cos(Th)/(1+ba)

       F(3) = cos(Th)
       F(4) = sin(Th)
       F(5) = 1.0

      END SUBROUTINE FUNC
!----------------------------------------------------------------------

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      REAL(8), PARAMETER :: PI = 4 * atan(1.0)
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

       PAR(1)=0.0
       PAR(2)=0.0
       PAR(3)=0.0

       U(1)=0.0
       U(2)=0.0

       U(3)=T
       U(4)=0.0
       U(5)=T

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
      FB(2)=U1(1)
      FB(3)=U0(2)+PAR(3)
      FB(4)=U0(3)
      FB(5)=U0(4)
      FB(6)=U0(5)

      END SUBROUTINE BCND
!----------------------------------------------------------------------
      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      !----------------------------------------------------------------------

      SUBROUTINE PVLS(NDIM,U,PAR)
      END SUBROUTINE PVLS

!----------------------------------------------------------------------
