#include "PROPOSAL/Propagator.h"
#include <iostream>
#include <math.h>

//#define CHECK_DISTANCE_FORMULAE

// trayectorias en esferas
static const double R = 6371324.0; // rock_ice boundary [meters]
static const double h = 882.0;     // center of detector to boundary [meters]

double GetDistanceCenterToRockIce(double cos_th){
    // change to fundamental earth facing frame
    // cos_th *= -1.;
    double sin_th = sqrt(1-cos_th*cos_th);
    if (sin_th > R/(R+h))
      return std::numeric_limits<double>::max();

    double sin_alpha = ((R+h)/R)*sin_th;
    double cos_alpha = sqrt(1-sin_alpha*sin_alpha);

    double cos_psi = sin_alpha*cos_th+sin_th*cos_alpha;
    double sin_psi = sin_th*sin_alpha-cos_alpha*cos_th;

    std::vector<double> P {R*cos_psi,R*sin_psi};

    return sqrt(pow(P[0]-0.,2)+pow(P[1]-(R+h),2));
}

vector<double> linspace(double Emin,double Emax,int div){
    vector<double> linpoints;
    double step_lin = (Emax - Emin)/double(div);

    double EE = Emin;
    while (EE <= Emax+0.001){
        linpoints.push_back(EE);
        EE = EE + step_lin;
    }

    return linpoints;
}

vector<double> logspace(double Emin,double Emax,int div){
    vector<double> logpoints;
    double Emin_log,Emax_log;
    if (Emin < 1.0e-5 ) {
        Emin_log = 0.0;
    } else {
        Emin_log = log(Emin);
    }
    Emax_log = log(Emax);

    double step_log = (Emax_log - Emin_log)/double(div);

    double EE = Emin_log;
    while (EE <= Emax_log+0.001){
        logpoints.push_back(exp(EE));
        EE = EE + step_log;
    }

    return logpoints;
}

using namespace std;

int main()
{
#ifdef CHECK_DISTANCE_FORMULAE
  // check distance formulae
  for(double cth : linspace(-1,0.1,20)){
    std::cout << cth << " " << GetDistanceCenterToRockIce(cth) << std::endl;
  }
  exit(0);
#endif

  const double GeV = 1.0e3;
  const double km = 1.0e5;
  const double meter = 1.0e2;

  EnergyCutSettings* ecut = new EnergyCutSettings(10,1e-3);
  Medium* ICE = new Medium("ice",1.);
  Propagator* ice_prop = new Propagator(ICE,ecut,"mu","resources/tables");//,false,true,true,true,1,12,1,1,1,1,false,2);
  Medium* ROCK = new Medium("standard_rock",1.);
  Propagator* rock_prop = new Propagator(ROCK,ecut,"mu","resources/tables");//,false,true,true,true,1,12,1,1,1,1,false,2);

  unsigned int number_of_particles = 1e4;
  unsigned int OnePercent = number_of_particles/100;

  std::vector<double> initial_energy_array = logspace(1.0e2*GeV,1.0e7*GeV,100); // in MeV
  std::vector<double> distance_array = linspace(0.1*km,10.0*km,100); // in cm

  for(double cth : linspace(-1,0.1,25)){
    double max_ice_distance = GetDistanceCenterToRockIce(cth);
    for( double initial_energy : initial_energy_array ){
      for( double distance : distance_array ){
        for( unsigned int i = 0; i < number_of_particles; i++ ){
          double ice_distance = max_ice_distance;
          double rock_distance = distance - max_ice_distance;
          if (rock_distance < 0 )
            rock_distance = 0;
            ice_distance = distance;

          // first rock_propagate in rock
          rock_prop->GetParticle()->SetEnergy(initial_energy);
          rock_prop->GetParticle()->SetPhi(0);
          rock_prop->GetParticle()->SetTheta(0);
          rock_prop->GetParticle()->SetX(0);
          rock_prop->GetParticle()->SetY(0);
          rock_prop->GetParticle()->SetZ(0);
          rock_prop->GetParticle()->SetT(0);
          rock_prop->GetParticle()->SetPropagatedDistance(0);
          rock_prop->GetParticle()->SetParticleId(0);

          double final_rock_energy  =   rock_prop->GetParticle()->GetEnergy();
          // then propagate in ice
          ice_prop->GetParticle()->SetEnergy(final_rock_energy);
          ice_prop->GetParticle()->SetPhi(0);
          ice_prop->GetParticle()->SetTheta(0);
          ice_prop->GetParticle()->SetX(0);
          ice_prop->GetParticle()->SetY(0);
          ice_prop->GetParticle()->SetZ(0);
          ice_prop->GetParticle()->SetT(0);
          ice_prop->GetParticle()->SetPropagatedDistance(0);
          ice_prop->GetParticle()->SetParticleId(0);

          double final_energy = ice_prop->GetParticle()->GetEnergy();

          std::cout << cth << " " << initial_energy << " " << distance << " " << final_energy << std::endl;
        }
      }
    }
  }

  return 0;
}
