import numpy as np
from scipy import integrate

class HertzClass():
        def __init__(self, inputs, materialparams):
            '''
            Hertzian class for calculating maximum contact pressure

            Parameters:
            inputs: [Rxw, Ryw, Rxr, Fn] in SI-units
            materialparams: [E_I, nu_I, E_II, nu_II] in SI-units

            Returns:
            A class for Hertzian theory
            '''
            self.Rxw = inputs[0]
            self.Ryw = inputs[1]
            self.Rxr = inputs[2]
            self.Fn = inputs[3]
            self.rhoxw = 1/self.Rxw
            self.rhoyw = 1/self.Ryw
            self.rhoxr = 1/self.Rxr
            self.rhoyr = 0 #Assume that rail has no curvature in y-z plane
            self.Ew = materialparams[0]
            self.nuw = materialparams[1]
            self.Er = materialparams[2]
            self.nur = materialparams[3]
        def f1(self,phi, kappa):
            '''The integrand of the first complete elliptical integral'''
            return (1-(1-1/kappa**2)*np.sin(phi)**2)**(-1/2)
        def f2(self,phi, kappa):
            '''The integrand of the second complete elliptical integral'''
            return (1-(1-1/kappa**2)*np.sin(phi)**2)**(1/2)
        def AB(self):
              '''
              Returns:
              A, B: Relative curvatures
              '''
              sumrho = self.rhoxw+self.rhoyw+self.rhoxr+self.rhoyr
              RI = self.rhoxw-self.rhoyw
              RII = self.rhoxr-self.rhoyr
              FP = -np.sqrt(RI**2+RII**2+2*RI*RII)/sumrho
              A = sumrho/4*(1+FP)
              B = sumrho/4*(1-FP)
              return A, B
        def fEstar(self):
              '''
              Returns:
              Estar: Equivalent Young's modulus
              '''
              Estar = 1/((1-self.nuw**2)/self.Ew+(1-self.nur**2)/self.Er)
              return Estar
        def kap(self):
              '''
              Returns:
              kappa: Ellipticity parameter
              Ekap: Second complete elliptic integral
              Fkap: First complete elliptic integral
              '''
              mu1 = 0.40227436
              mu2 = 3.7491752e-2
              mu3 = 7.4855761e-4
              mu4 = 2.1667028e-6
              mu5 = 0.42678878
              mu6 = 4.2605401e-2
              mu7 = 9.0786922e-4
              mu8 = 2.7868927e-6
              A, B = HertzClass.AB(self)
              X = np.log10(B/A)
              gamma = 2/3*(1+mu1*X**2+mu2*X**4+mu3*X**6+mu4*X**8)/(1+mu5*X**2+mu6*X**4+mu7*X**6+mu8*X**8)
              kappa = (B/A)**gamma
              phivec = np.linspace(0,np.pi/2, 10000)
              f1_vec = HertzClass.f1(self,phivec, kappa)
              f2_vec = HertzClass.f2(self,phivec, kappa)
              Ekap = integrate.trapezoid(f2_vec, phivec)
              Fkap = integrate.trapezoid(f1_vec, phivec)
              return kappa, Ekap, Fkap
        def predic(self):
              '''
              Returns:
              pmax: Maximum contact pressure in Pa
              '''
              A, B = HertzClass.AB(self)
              kappa, Ekap, Fkap = HertzClass.kap(self)
              e = np.sqrt(1-1/kappa**2)
              Estar = HertzClass.fEstar(self)
              e = np.sqrt(1-1/kappa**2)
              a = (3*self.Fn*(Fkap-Ekap)/2/np.pi/e**2/Estar/A)**(1/3)
              b = a*np.sqrt(1-e**2)
              pmax = 3*self.Fn/2/np.pi/a/b#
              return pmax