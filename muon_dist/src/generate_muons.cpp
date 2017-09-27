#include "PROPOSAL/Propagator.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <math.h>

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

int main(int argc, char * argv[])
{
    const double GeV = 1.0e3;
    const double km = 1.0e5;
    const double meter = 1.0e2;

    double energy;
    std::stringstream ss;
    std::string s(argv[1]);
    ss.str(s);
    ss >> energy;

    int seed;
    ss.clear();
    s = std::string(argv[2]);
    ss.str(s);
    ss >> seed;

    int n_muons;
    ss.clear();
    s = std::string(argv[3]);
    ss.str(s);
    ss >> n_muons;

    EnergyCutSettings* ecut = new EnergyCutSettings(10,1e-4);
    Medium* ICE = new Medium("ice",1.);
    Propagator* ice_prop = new Propagator(ICE,ecut,"mu","resources/tables");
    ice_prop->set_seed(seed);

    ss.str("");
    ss.clear();
    ss << "./" << energy;
    std::string dir_name = ss.str();
    mkdir(dir_name.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP | S_IROTH);
    ss <<"/" << seed << ".txt";
    std::string f_name = ss.str();

    std::cout << f_name << std::endl;

    std::fstream f(f_name, std::fstream::out);

    for(int i=0; i<n_muons; ++i)
    {
        ice_prop->GetParticle()->SetEnergy(energy);
        ice_prop->GetParticle()->SetPhi(0);
        ice_prop->GetParticle()->SetTheta(0);
        ice_prop->GetParticle()->SetX(0);
        ice_prop->GetParticle()->SetY(0);
        ice_prop->GetParticle()->SetZ(0);
        ice_prop->GetParticle()->SetT(0);
        ice_prop->GetParticle()->SetPropagatedDistance(0);
        ice_prop->GetParticle()->SetParticleId(0);

        ice_prop->Propagate(1e9);

        double final_energy = ice_prop->GetParticle()->GetEnergy();

        double final_distance = ice_prop->GetParticle()->GetPropagatedDistance();

        f << final_distance << std::endl;
    }

}

