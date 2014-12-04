/*
  example program how to read
  xmaxHistograms.txt

  Load via
  .L example.C+
  and then execute via
  example()
  at the ROOT prompt

*/

#include <TH1D.h>
#include <TF1.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TROOT.h>
#include <TMath.h>
#include <TCanvas.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
using namespace std;

double resolutionFunc(double* , double*);
double acceptanceFunc(double* , double*);
double fullWidth(const TF1*);

void
example()
{

  gStyle->SetOptTitle(1);
  gStyle->SetOptStat(1111);
  gROOT->SetStyle("Plain");

  //----------------------------------------------------------
  // read Xmax histograms from 'xmaxHistograms.txt'

  vector<TH1D*> xmaxDistributions;
  vector<double> minLgE;
  vector<double> maxLgE;
  fstream histogramFile("xmaxHistograms.txt");
  while (true) {
    int iEnergy, nEntries, nBins;
    double lgEmin, lgEmax, xMin, xMax;
    histogramFile >> iEnergy >> lgEmin >> lgEmax
                  >> nEntries >> nBins >> xMin >> xMax;
    if (!histogramFile.good())
      break;
    ostringstream histName;
    histName << "xmaxDistribution" << iEnergy;
    ostringstream histTitle;
    histTitle << fixed << setprecision(1) << showpoint
              << lgEmin << " < lg(E/eV) < " << lgEmax
              << "; X_{max} [g/cm^{2}]; entries";
    TH1D* xmaxHist = new TH1D(histName.str().c_str(),
                              histTitle.str().c_str(),
                              nBins, xMin, xMax);
    for (int iBin = 0; iBin < nBins; ++iBin) {
      double content;
      histogramFile >> content;
      xmaxHist->SetBinContent(iBin + 1, content);
    }
    xmaxHist->SetEntries(nEntries);
    xmaxDistributions.push_back(xmaxHist);
    minLgE.push_back(lgEmin);
    maxLgE.push_back(lgEmax);
  }

  if (xmaxDistributions.empty()) {
    cerr << " error reading xmaxHistograms.txt" << endl;
    return;
  }

  //----------------------------------------------------------
  // read resolution parameters from 'resolution.txt'

  vector<TF1*> xmaxResolutionsDefault;
  vector<TF1*> xmaxResolutionsUp;
  vector<TF1*> xmaxResolutionsLow;
  fstream resolutionFile("resolution.txt");
  while (true) {
    int iEnergy;
    double lgEmin, lgEmax, sigma1, sigma1Err,
      sigma2, sigma2Err, fraction;
    resolutionFile >> iEnergy >> lgEmin >> lgEmax
                   >> sigma1 >> sigma1Err
                   >> sigma2 >> sigma2Err
                   >> fraction;
    if (!resolutionFile.good())
      break;

    ostringstream funcTitle;
    funcTitle << "; #DeltaX_{max} [g/cm^{2}]; probability";

    ostringstream funcNameDefault;
    funcNameDefault << "resolutionDefault" << iEnergy;
    TF1* resoFuncDefault = new TF1(funcNameDefault.str().c_str(), resolutionFunc,
                                   -65, 65, 3);
    resoFuncDefault->SetParameters(sigma1, sigma2, fraction);
    resoFuncDefault->SetLineColor(kRed);
    resoFuncDefault->SetLineWidth(1);
    resoFuncDefault->SetTitle(funcTitle.str().c_str());

    ostringstream funcNameUp;
    funcNameUp << "resolutionUp" << iEnergy;
    TF1* resoFuncUp = new TF1(funcNameUp.str().c_str(), resolutionFunc,
                              -65, 65, 3);
    resoFuncUp->SetParameters(sigma1 + sigma1Err, sigma2 + sigma2Err, fraction);
    resoFuncUp->SetLineColor(kBlack);
    resoFuncUp->SetLineWidth(1);
    resoFuncUp->SetLineStyle(2);
    resoFuncUp->SetTitle(funcTitle.str().c_str());

    ostringstream funcNameLow;
    funcNameLow << "resolutionLow" << iEnergy;
    TF1* resoFuncLow = new TF1(funcNameLow.str().c_str(), resolutionFunc,
                               -65, 65, 3);
    resoFuncLow->SetParameters(sigma1 - sigma1Err, sigma2 - sigma2Err, fraction);
    resoFuncLow->SetLineColor(kBlack);
    resoFuncLow->SetLineWidth(1);
    resoFuncLow->SetLineStyle(2);
    resoFuncLow->SetTitle(funcTitle.str().c_str());

    xmaxResolutionsDefault.push_back(resoFuncDefault);
    xmaxResolutionsUp.push_back(resoFuncUp);
    xmaxResolutionsLow.push_back(resoFuncLow);
  }

  if (xmaxResolutionsUp.size() != xmaxDistributions.size()) {
    cerr << " error reading resolution.txt" << endl;
    return;
  }


  //----------------------------------------------------------
  // read acceptance parameters from 'acceptance.txt'

  vector<TF1*> xmaxAcceptances;
  vector<TF1*> xmaxAcceptancesSys1;
  vector<TF1*> xmaxAcceptancesSys2;
  fstream acceptanceFile("acceptance.txt");
  while (true) {
    int iEnergy;
    double lgEmin, lgEmax, x1, ex1, x2, ex2, l1, el1, l2, el2;
    acceptanceFile >> iEnergy >> lgEmin >> lgEmax
                   >> x1 >> ex1 >> x2 >> ex2
                   >> l1 >> el1 >> l2 >> el2;
    if (!acceptanceFile.good())
      break;
    ostringstream funcName;
    funcName << "acceptance" << iEnergy;
    ostringstream funcTitle;
    funcTitle << "; #DeltaX_{max} [g/cm^{2}]; relative acceptance";

    TF1* accFunc = new TF1(funcName.str().c_str(), acceptanceFunc,
                           200, 1500, 4);
    accFunc->SetParameters(x1, x2, l1, l2);
    accFunc->SetLineColor(kBlue);
    accFunc->SetLineWidth(1);
    accFunc->SetTitle(funcTitle.str().c_str());
    xmaxAcceptances.push_back(accFunc);

    funcName.str("");
    funcName << "acceptance" << iEnergy << "sys1";
    TF1* accFuncSys1 = new TF1(funcName.str().c_str(), acceptanceFunc,
                               200, 1500, 4);
    accFuncSys1->SetParameters(x1 - ex1, x2 + ex2, l1 + el1, l2 + el2);
    accFuncSys1->SetLineWidth(1);
    accFuncSys1->SetLineStyle(2);
    accFuncSys1->SetTitle(funcTitle.str().c_str());
    xmaxAcceptancesSys1.push_back(accFuncSys1);

    funcName.str("");
    funcName << "acceptance" << iEnergy << "sys1";
    TF1* accFuncSys2 = new TF1(funcName.str().c_str(), acceptanceFunc,
                               200, 1500, 4);
    accFuncSys2->SetParameters(x1 + ex1, x2 - ex2, l1 - el1, l2 - el2);
    accFuncSys2->SetLineWidth(1);
    accFuncSys2->SetLineStyle(2);
    accFuncSys2->SetTitle(funcTitle.str().c_str());
    xmaxAcceptancesSys2.push_back(accFuncSys2);

  }

  if (xmaxAcceptances.size() != xmaxDistributions.size()) {
    cerr << " error reading acceptance.txt" << endl;
    return;
  }


  //----------------------------------------------------------
  // read resolution parameters from 'resolution.txt'

  vector<double> xmaxSysUp;
  vector<double> xmaxSysLow;
  fstream systematicsFile("xmaxSystematics.txt");
  while (true) {
    int iEnergy;
    double lgEmin, lgEmax, sigmaUp, sigmaLow;
    systematicsFile >> iEnergy >> lgEmin >> lgEmax
                    >> sigmaUp >> sigmaLow;
    if (!systematicsFile.good())
      break;
    xmaxSysUp.push_back(sigmaUp);
    xmaxSysLow.push_back(sigmaLow);
  }

  if (xmaxSysLow.size() != xmaxDistributions.size()) {
    cerr << " error reading xmaxSystematics.txt" << endl;
    return;
  }


  //----------------------------------------------------------
  // create canvas with three sub-canvases

  TCanvas* canvas = new TCanvas("xmax", "", 450, 700);
  canvas->Divide(1, 2);
  TVirtualPad* xmaxPad = canvas->cd(1);
  TVirtualPad* accResoPad = canvas->cd(2);
  accResoPad->Divide(1, 2);
  TVirtualPad* accPad = accResoPad->cd(1);
  TVirtualPad* resoPad = accResoPad->cd(2);
  accPad->SetBottomMargin(0.2);
  resoPad->SetBottomMargin(0.2);
  accPad->SetTopMargin(0.02);
  resoPad->SetTopMargin(0.02);

  //----------------------------------------------------------
  // ask for energy and draw Xmax distribution,
  // resolution and acceptance

  while (true) {
    cout << "\n --> Please enter lg(E/eV) to display (< 0 to quit)" << endl;
    double lgE;
    cin >> lgE;
    if (!cin.good()) {
      cerr << " illegal input" << endl;
      cin.clear(); cin.ignore();
      continue;
    }
    else if (lgE < 0)
      break;
    else if (lgE < minLgE.front() || lgE >= maxLgE.back()) {
      cerr << " lg(E/eV) = " << lgE << " is out of range ["
           << minLgE.front() << ", " << maxLgE.back() << "[, please re-enter!"
           << endl;
      continue;
    }

    int index = -1;
    for (unsigned int iEnergy = 0; iEnergy < minLgE.size(); ++iEnergy) {
      if (minLgE[iEnergy] <= lgE && maxLgE[iEnergy] > lgE) {
        index = iEnergy;
        break;
      }
    }

    TLatex l;
    l.SetTextAlign(23);
    l.SetTextSize(0.04);
    l.SetTextColor(kBlack);
    l.SetNDC(true);

    //-----------------------
    xmaxPad->cd();
    const double axisMax = 1150; // deepest shower is at 1140 g/cm^2
    const double axisMin =
      2 * xmaxDistributions[index]->GetMean() - axisMax;
    xmaxDistributions[index]->GetXaxis()->SetRangeUser(axisMin, axisMax);
    xmaxDistributions[index]->Draw();
    ostringstream xmaxSys;
    xmaxSys << showpoint << fixed << setprecision(1)
            << "#sigma(X_{max})_{sys} = ^{+" << xmaxSysUp[index]
            << "}_{-" << xmaxSysLow[index] << "} g/cm^{2} ";
    l.DrawLatex(0.7, 0.7, xmaxSys.str().c_str());

    //-----------------------
    const double factor = 1.6;
    const double labelXSize = gStyle->GetLabelSize("X") * factor;
    const double labelYSize = gStyle->GetLabelSize("Y") * factor;
    const double titleXSize = gStyle->GetTitleSize("X") * factor;
    const double titleYSize = gStyle->GetTitleSize("Y") * factor;
    resoPad->cd();
    xmaxResolutionsLow[index]->Draw();
    xmaxResolutionsLow[index]->GetXaxis()->SetLabelSize(labelXSize);
    xmaxResolutionsLow[index]->GetYaxis()->SetLabelSize(labelYSize);
    xmaxResolutionsLow[index]->GetXaxis()->SetTitleSize(titleXSize);
    xmaxResolutionsLow[index]->GetYaxis()->SetTitleSize(titleYSize);
    xmaxResolutionsLow[index]->GetYaxis()->SetTitleOffset(0.8);
    xmaxResolutionsUp[index]->Draw("SAME");
    xmaxResolutionsDefault[index]->Draw("SAME");
    cout << xmaxResolutionsLow[index]->Integral(-200, 200)
         << xmaxResolutionsUp[index]->Integral(-200, 200) << endl;
    ostringstream resolution;
    resolution << "RMS = "
               << showpoint << fixed << setprecision(1)
               << fullWidth(xmaxResolutionsDefault[index])
               << "#pm"
               << (fullWidth(xmaxResolutionsUp[index]) -
                   fullWidth(xmaxResolutionsLow[index]))/2.
               << " g/cm^{2}";
    l.SetTextSize(0.06);
    l.DrawLatex(0.75, 0.93, resolution.str().c_str());

    //-----------------------
    accPad->cd();
    xmaxAcceptances[index]->GetXaxis()->SetLabelSize(labelXSize);
    xmaxAcceptances[index]->GetYaxis()->SetLabelSize(labelYSize);
    xmaxAcceptances[index]->GetXaxis()->SetTitleSize(titleXSize);
    xmaxAcceptances[index]->GetYaxis()->SetTitleSize(titleYSize);
    xmaxAcceptances[index]->GetYaxis()->SetRangeUser(0, 1.3);
    xmaxAcceptances[index]->GetYaxis()->SetTitleOffset(0.8);
    xmaxAcceptances[index]->GetXaxis()->SetRangeUser(axisMin, axisMax);
    xmaxAcceptances[index]->Draw();
    xmaxAcceptancesSys1[index]->Draw("SAME");
    xmaxAcceptancesSys2[index]->Draw("SAME");

    canvas->Update();
  }

}


// double Gauss
double
resolutionFunc(double* x, double* p)
{
  return p[2] * TMath::Gaus(*x, 0, p[0], true) +
    (1 - p[2]) * TMath::Gaus(*x, 0, p[1], true);
}

// full width of double Gauss
double
fullWidth(const TF1* func)
{
  const double s1 = func->GetParameter(0);
  const double s2 = func->GetParameter(1);
  const double f = func->GetParameter(2);
  const double fullVariance = f * pow(s1, 2) +
    (1 - f) * pow(s2, 2);
  return sqrt(fullVariance);
}

// acceptance function
double
acceptanceFunc(double* x, double* p)
{
  const double X = *x;
  if (X < p[0])
    return exp((X-p[0])/p[2]);
  else if (X < p[1])
    return 1;
  else
    return exp(-(X-p[1])/p[3]);
}


