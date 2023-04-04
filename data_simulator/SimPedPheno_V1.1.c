#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#define nMaxIndPerGen 100000
#define nMaxGen 10
#define nMaxTrait 10

gsl_rng * r;
gsl_matrix * Sigma;
gsl_matrix * Liab;
gsl_matrix * FstGenLiab;
gsl_matrix * tmpLiab;
gsl_matrix * tmpSigma;

double * HeritL[nMaxGen][nMaxTrait];
long int * Famlist[nMaxGen][4]; // 10 generations * 3 members (0 - mo, 1 - fa, 2 - offspring) + offspring gender * nIndividual per generation
long int Trio[3]; // Current trio to be recorded (0 - mo, 1 - fa, 3 - offspring)
long int nIndividual; // Individual counter for certain generation
int nGen; // Generation counter
double bAge[nMaxTrait], bSex[nMaxTrait], bHerr[nMaxTrait], bEnv[nMaxTrait]; // coefficients
double LiabThres[nMaxTrait];
double ParW[nMaxTrait][2]; //maternal effect propotion and paternal effect propotion, sum = 1
double HerrCoeff[nMaxTrait]; // A coefficient add to non-zero generation genetic risk so that it's variance does not shrink anymore 
char buffer[500000];
char AgeDoILiabCol[200];
char Age10YLiabCol[200];
char GenderLiabCol[200];
char HeritLiabCol[200];
char EnvLiabCol[200];
char LiabDoICol[200];
char Liab10YCol[200];
char EndPtDoICol[200];
char EndPt10YCol[200];

//Parameters
long int nIndPerGen;
int nTotGen;
int nTotTrait;
double AgeRange[2];
double pAge[nMaxTrait], pSex[nMaxTrait], pHerr[nMaxTrait], pEnv[nMaxTrait]; // variance propotion, need to check distribution variance
double prev[nMaxTrait];
double TraitCorr[nMaxTrait];
char Output[1000];



// Need to fix the parameter loading (for multitrait)
void ReadParam(const char *ParIn) {
	FILE *ParFile;
	int i, j;
	char *tok; char *p;
	char *tok2; char *p2;
	char tmp[200];
	ParFile = fopen(ParIn,"r");
	// Set default values
	memset(pAge, 0.0, sizeof(double) * nMaxTrait);
	memset(pSex, 0.0, sizeof(double) * nMaxTrait);
	memset(pHerr, 0.5, sizeof(double) * nMaxTrait);
	memset(pEnv, 0.5, sizeof(double) * nMaxTrait);
	memset(prev, 0.2, sizeof(double) * nMaxTrait);
	memset(TraitCorr, 0, sizeof(double) * nMaxTrait);
	memset(ParW, 0.5, sizeof(double) * 2);
	memset(HerrCoeff, sqrt(1.0/(pow(0.5, 2) + pow(0.5, 2))), sizeof(double) * nMaxTrait);
	TraitCorr[0] = 1.0;
	nTotGen = 3;
	nTotTrait = 1;
	nIndPerGen = 100;
	AgeRange[0] = 18;
	AgeRange[1] = 80;

	strcpy(Output, "OutPedigree");

	if (ParFile == NULL) {
	    printf("Cannot open parameter file.\n");
	    exit(0);
	}
	else {
		while (fgets(buffer, sizeof(buffer), ParFile) != NULL) {
			p = buffer;
			tok = strtok_r(p, " \t", &p);
	    	if (tok != NULL) {
	    		if (strcmp(tok, "AgeLower") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			AgeRange[0] = atof(tok);
	    			if (AgeRange[0] <= 0) {
	    				printf("AgeLower should be positive.\n");
						exit(0);
	    			}
				}
				if (strcmp(tok, "AgeUpper") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			AgeRange[1] = atof(tok);
	    			if (AgeRange[1] <= 0) {
	    				printf("AgeUpper should be positive.\n");
						exit(0);
	    			}
				}
				if (strcmp(tok, "MaternalWeight") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			strcpy(tmp, tok);
	    			p2 = tmp;
	    			i = 0;
	    			while ((tok2 = strtok_r(p2, ",", &p2))) {
				    	if (i < nMaxTrait) {
				    		if ( (atof(tok2) >= 0) && (atof(tok2) <= 1) ) {
					    		ParW[i][0] = atof(tok2);
					    		ParW[i][1] = 1.0 - ParW[i][0];
				    		}
				    		else {
				    			printf("MaternalWeight should be between 0 to 1. PaternalWeight = 1 - MaternalWeight.\n");
				    			exit(0);
				    		}
				    	}
				    	else {
				    		printf("Too many MaternalWeight input!\n");
				    		exit(0);
				    	}
				    	i++;
					}
					if ((nTotTrait == 1) && (i != 1))
						nTotTrait = i;
					else if (nTotTrait != i) {
						printf("Wrong number of MaternalWeight input!\n");
				    	exit(0);
					}
				}
	    		if (strcmp(tok, "Prevalence") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			strcpy(tmp, tok);
	    			p2 = tmp;
	    			i = 0;
	    			while ((tok2 = strtok_r(p2, ",", &p2))) {
				    	if (i < nMaxTrait) {
				    		if ( (atof(tok2) > 0) && (atof(tok2) < 1) ) 
					    		prev[i] = atof(tok2);
				    		else {
				    			printf("Prevalence should be between 0 to 1.\n");
				    			exit(0);
				    		}
				    	}
				    	else {
				    		printf("Too many Prevalence input!\n");
				    		exit(0);
				    	}
				    	i++;
					}
					if ((nTotTrait == 1) && (i != 1))
						nTotTrait = i;
					else if (nTotTrait != i) {
						printf("Wrong number of Prevalence input!\n");
				    	exit(0);
					}
				}
				if (strcmp(tok, "AgeEffect") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			strcpy(tmp, tok);
	    			p2 = tmp;
	    			i = 0;
	    			while ((tok2 = strtok_r(p2, ",", &p2))) {
				    	if (i < nMaxTrait) {
				    		if ( (atof(tok2) >= 0) && (atof(tok2) <= 1) )
					    		pAge[i] = atof(tok2);
				    		else {
				    			printf("Propotion of AgeEffect should be between 0 to 1.\n");
				    			exit(0);
				    		}
				    	}
				    	else {
				    		printf("Too many AgeEffect input!\n");
				    		exit(0);
				    	}
				    	i++;
					}
					if ((nTotTrait == 1) && (i != 1))
						nTotTrait = i;
					else if (nTotTrait != i) {
						printf("Wrong number of AgeEffect input!\n");
				    	exit(0);
					}
				}
				if (strcmp(tok, "SexEffect") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			strcpy(tmp, tok);
	    			p2 = tmp;
	    			i = 0;
	    			while ((tok2 = strtok_r(p2, ",", &p2))) {
				    	if (i < nMaxTrait) {
				    		if ( (atof(tok2) >= 0) && (atof(tok2) <= 1) )
					    		pSex[i] = atof(tok2);
				    		else {
				    			printf("Propotion of SexEffect should be between 0 to 1.\n");
				    			exit(0);
				    		}
				    	}
				    	else {
				    		printf("Too many SexEffect input!\n");
				    		exit(0);
				    	}
				    	i++;
					}
					if ((nTotTrait == 1) && (i != 1))
						nTotTrait = i;
					else if (nTotTrait != i) {
						printf("Wrong number of SexEffect input!\n");
				    	exit(0);
					}
				}
				if (strcmp(tok, "Herritable") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			strcpy(tmp, tok);
	    			p2 = tmp;
	    			i = 0;
	    			while ((tok2 = strtok_r(p2, ",", &p2))) {
				    	if (i < nMaxTrait) {
				    		if ( (atof(tok2) >= 0) && (atof(tok2) <= 1) )
				    			pHerr[i] = atof(tok2);
				    		else {
				    			printf("Propotion of Herritable effect should be between 0 to 1.\n");
				    			exit(0);
				    		}
				    	}
				    	else {
				    		printf("Too many Herritable effect input!\n");
				    		exit(0);
				    	}
				    	i++;
					}
					if ((nTotTrait == 1) && (i != 1))
						nTotTrait = i;
					else if (nTotTrait != i) {
						printf("Wrong number of Herritable effect input!\n");
				    	exit(0);
					}
				}
				if (strcmp(tok, "TraitCorrelation") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			strcpy(tmp, tok);
	    			p2 = tmp;
	    			i = 0;
	    			while ((tok2 = strtok_r(p2, ",", &p2))) {
				    	if (i < nMaxTrait) {
				    		if ( (atof(tok2) >= -1) && (atof(tok2) <= 1) )
					    		TraitCorr[i] = atof(tok2);
				    		else {
				    			printf("TraitCorrelation should be between -1 and 1.\n");
				    			exit(0);
				    		}
				    	}
				    	else {
				    		printf("Too many TraitCorrelation input!\n");
				    		exit(0);
				    	}
				    	i++;
					}
					if ((nTotTrait == 1) && (i != 1))
						nTotTrait = i;
					else if (nTotTrait != i) {
						printf("Wrong number of TraitCorrelation input!\n");
				    	exit(0);
					}
				}
	    		else if (strcmp(tok, "nGeneration") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			nTotGen = atoi(tok);
	    			if ((nTotGen <= 0) || (nTotGen > nMaxGen)) {
	    				printf("Number of generation should be between 1 to %d\n", nMaxGen);
						exit(0);
	    			}
	    		}
	    		else if (strcmp(tok, "nIndividual") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			nIndPerGen = atoi(tok);
	    			if ((nIndPerGen < 20) || (nIndPerGen > nMaxIndPerGen)) {
	    				printf("Number of individuals per generation should be between 20 to %d\n", nMaxIndPerGen);
						exit(0);
	    			}
	    		}
	    		else if (strcmp(tok, "OutPrefix") == 0) {
	    			tok = strtok_r(p, " \t\n", &p);
	    			strcpy(Output, tok);
	    		}
	    	}
		}
	}

	// Small checks
	if (AgeRange[0] >= AgeRange[1]) {
		printf("AgeUpper should be higher than AgeLower, reversing age bounds.\n");
		double tmp;
		tmp = AgeRange[0];
		AgeRange[0] = AgeRange[1];
		AgeRange[1] = tmp;
	}
	for (i = 0; i < nTotTrait; i++) {
		pEnv[i] = 1 - (pAge[i] + pSex[i] + pHerr[i]);
		if (pEnv[i] <= 0) {
			printf("Summation of Age, Sex and Heritable effect should be less than 1 for the %d-th trait.\n", i+1);
			exit(0);
		}
	}
	if (TraitCorr[0] != 1.0) {
		printf("TraitCorr uses first trait as reference. Making the first correlation coefficient 1.0. \n");
		TraitCorr[0] = 1.0;
	}

	// allocate memory and Initialize 
	for (i = 0; i < nMaxGen; i++) {
		for (j = 0; j < nMaxTrait; j++) {
			HeritL[i][j] = malloc(sizeof(double) * nIndPerGen);
			memset(HeritL[i][j], 0.0, sizeof(double) * nIndPerGen);
		}
	}
	
	for (i = 0; i < nMaxGen; i++) {
		for (j = 0; j < 4; j++) {
			Famlist[i][j] = malloc(sizeof(long int) * nIndPerGen);
			memset(Famlist[i][j], 0, sizeof(long int) * nIndPerGen);
		}
	}

	for (i = 0; i < nTotTrait; i++) {
		sprintf(tmp, "AgeDoILiab-%d,", i+1);
		strcat(AgeDoILiabCol, tmp);
		sprintf(tmp, "Age10YLiab-%d,", i+1);
		strcat(Age10YLiabCol, tmp);
		sprintf(tmp, "GenderLiab-%d,", i+1);
		strcat(GenderLiabCol, tmp);
		sprintf(tmp, "HeritLiab-%d,", i+1);
		strcat(HeritLiabCol, tmp);
		sprintf(tmp, "EnvLiab-%d,", i+1);
		strcat(EnvLiabCol, tmp);
		sprintf(tmp, "LiabDoI-%d,", i+1);
		strcat(LiabDoICol, tmp);
		sprintf(tmp, "Liab10Y-%d,", i+1);
		strcat(Liab10YCol, tmp);
		sprintf(tmp, "EndPtDoI-%d,", i+1);
		strcat(EndPtDoICol, tmp);
		sprintf(tmp, "EndPt10Y-%d,", i+1);
		strcat(EndPt10YCol, tmp);

		HerrCoeff[i] = sqrt(1.0/(pow(ParW[i][0], 2) + pow(ParW[i][1], 2)));
	}

	AgeDoILiabCol[strlen(AgeDoILiabCol)-1] = '\0';
	Age10YLiabCol[strlen(Age10YLiabCol)-1] = '\0';
	GenderLiabCol[strlen(GenderLiabCol)-1] = '\0';
	HeritLiabCol[strlen(HeritLiabCol)-1] = '\0';
	EnvLiabCol[strlen(EnvLiabCol)-1] = '\0';
	LiabDoICol[strlen(LiabDoICol)-1] = '\0';
	Liab10YCol[strlen(Liab10YCol)-1] = '\0';
	EndPtDoICol[strlen(EndPtDoICol)-1] = '\0';
	EndPt10YCol[strlen(EndPt10YCol)-1] = '\0';

	FILE *OutFile;
	OutFile = fopen(Output,"w");
	if (OutFile == NULL) {
        printf("Cannot open output file.\n");
        exit(0);
    } //check first wether the output file can be opened
	fprintf(OutFile, "Mother\tFather\tOffspring\tAgeDoI\t%s\t%s\tGender\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", AgeDoILiabCol, Age10YLiabCol, GenderLiabCol, HeritLiabCol, EnvLiabCol, LiabDoICol, Liab10YCol, EndPtDoICol, EndPt10YCol);
	fclose(OutFile);

	printf("Read parameters, done.\n");
	printf("Simulating %d generations, each with %ld individuals and %d traits.\n", nTotGen, nIndPerGen, nTotTrait);
}




// Need to double check
// from propotions get coefficients and prevelence liability cutoff
void GetBeta() {
	double mu;
	int i, j;

	for (i = 0; i < nTotTrait; i++) {
		bHerr[i] = sqrt(pHerr[i]);
		bEnv[i] = sqrt(pEnv[i]);
		bAge[i] = sqrt(12*pAge[i]);
		bSex[i] = sqrt(4*pSex[i]);
		mu = 0.5 * bSex[i] + 0.5 * bAge[i];
		LiabThres[i] = gsl_cdf_gaussian_Qinv(prev[i], sqrt(1.0)) + mu;
		printf("Trait %d out of %d:\n", i+1, nTotTrait);
		printf("bHerr = %lf, bEnv = %lf, bAge = %lf, bSex = %lf.\n", bHerr[i], bEnv[i], bAge[i], bSex[i]);
		printf("Liability threshold under given prevalence is %lf.\n", LiabThres[i]);
	}
	printf("Generate coefficients, done.\n");
}



void MakeCovMat() {
	int i;
	int status;
	Sigma = gsl_matrix_calloc(nTotTrait, nTotTrait);
	gsl_matrix_set_zero(Sigma);
	for (i = 0; i < nTotTrait; i++) {
		gsl_matrix_set(Sigma, i, i, 1.0);
		if (i != 0) {
			gsl_matrix_set(Sigma, 0, i, TraitCorr[i]);
			gsl_matrix_set(Sigma, i, 0, TraitCorr[i]);
		}
	}
	status = gsl_linalg_cholesky_decomp1(Sigma);

	if (!status) {
		printf("Trait correlation matrix is positive definite, ok!\n");
	}
	else {
		printf("Trait correlation matrix is not positive definite, assuming 0 correlation across traits.\n");
		gsl_matrix_set_zero(Sigma);
		for (i = 0; i < nTotTrait; i++)
			gsl_matrix_set(Sigma, i, i, 1.0);
	}
	Liab = gsl_matrix_calloc(nTotTrait, nIndPerGen);
	FstGenLiab = gsl_matrix_calloc(nTotTrait, nIndPerGen);
	tmpLiab = gsl_matrix_calloc(nTotTrait, nIndPerGen);
	tmpSigma = gsl_matrix_calloc(nTotTrait, nTotTrait);
}



void WriteRes(double Age, int Gender, double HeritLiab[nMaxTrait], double EnvLiab[nMaxTrait], double LiabDoI[nMaxTrait], double Liab10Y[nMaxTrait]) {
	int EndPtDoI[nMaxTrait], EndPt10Y[nMaxTrait];
	double AgeDoI;
	int i;
	char tmp[10];
	char tmpAgeDoILiab[200];
	char tmpAge10YLiab[200];
	char tmpGenderLiab[200];
	char tmpHeritLiab[200];
	char tmpEnvLiab[200];
	char tmpLiabDoI[200];
	char tmpLiab10Y[200];
	char tmpEndPtDoI[200];
	char tmpEndPt10Y[200];

	memset(EndPtDoI, 0, sizeof(int) * nMaxTrait);
	memset(EndPt10Y, 0, sizeof(int) * nMaxTrait);
	memset(tmpAgeDoILiab, '\0', sizeof tmpAgeDoILiab);
    memset(tmpAge10YLiab, '\0', sizeof tmpAge10YLiab);
    memset(tmpGenderLiab, '\0', sizeof tmpGenderLiab);
	memset(tmpHeritLiab, '\0', sizeof tmpHeritLiab);
    memset(tmpEnvLiab, '\0', sizeof tmpEnvLiab);
    memset(tmpLiabDoI, '\0', sizeof tmpLiabDoI);
    memset(tmpLiab10Y, '\0', sizeof tmpLiab10Y);
    memset(tmpEndPtDoI, '\0', sizeof tmpEndPtDoI);
    memset(tmpEndPt10Y, '\0', sizeof tmpEndPt10Y);

	AgeDoI = AgeRange[0] + Age*(AgeRange[1]-AgeRange[0]);
	
	for (i = 0; i < nTotTrait; i++) {
		EndPtDoI[i] = (LiabDoI[i] > LiabThres[i] ? 1 : 0);
		EndPt10Y[i] = (Liab10Y[i] > LiabThres[i] ? 1 : 0);

		sprintf(tmp, "%.4f,", bAge[i]*Age);
		strcat(tmpAgeDoILiab, tmp);
		sprintf(tmp, "%.4f,", bAge[i]*(Age+10/(AgeRange[1]-AgeRange[0])));
		strcat(tmpAge10YLiab, tmp);
		sprintf(tmp, "%.4f,", bSex[i]*Gender);
		strcat(tmpGenderLiab, tmp);
		sprintf(tmp, "%.4f,", bHerr[i]*HeritLiab[i]);
		strcat(tmpHeritLiab, tmp);
		sprintf(tmp, "%.4f,", bEnv[i]*EnvLiab[i]);
		strcat(tmpEnvLiab, tmp);
		sprintf(tmp, "%.4f,", LiabDoI[i]);
		strcat(tmpLiabDoI, tmp);
		sprintf(tmp, "%.4f,", Liab10Y[i]);
		strcat(tmpLiab10Y, tmp);
		sprintf(tmp, "%d,", EndPtDoI[i]);
		strcat(tmpEndPtDoI, tmp);
		sprintf(tmp, "%d,", EndPt10Y[i]);
		strcat(tmpEndPt10Y, tmp);
	}

	tmpAgeDoILiab[strlen(tmpAgeDoILiab)-1] = '\0';
	tmpAge10YLiab[strlen(tmpAge10YLiab)-1] = '\0';
	tmpGenderLiab[strlen(tmpGenderLiab)-1] = '\0';
	tmpHeritLiab[strlen(tmpHeritLiab)-1] = '\0';
	tmpEnvLiab[strlen(tmpEnvLiab)-1] = '\0';
	tmpLiabDoI[strlen(tmpLiabDoI)-1] = '\0';
	tmpLiab10Y[strlen(tmpLiab10Y)-1] = '\0';
	tmpEndPtDoI[strlen(tmpEndPtDoI)-1] = '\0';
	tmpEndPt10Y[strlen(tmpEndPt10Y)-1] = '\0';

	FILE *OutFile;
	OutFile = fopen(Output,"a");
	fprintf(OutFile, "%ld\t%ld\t%ld\t%.2f\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", 
		Trio[0], Trio[1], Trio[2], AgeDoI, tmpAgeDoILiab, tmpAge10YLiab, Gender, tmpGenderLiab, tmpHeritLiab, tmpEnvLiab, tmpLiabDoI, tmpLiab10Y, tmpEndPtDoI, tmpEndPt10Y);
	fclose(OutFile);
}



void LiabGenerator() {
	long int i, j;
	gsl_matrix_set_zero(Liab);
	for (i = 0; i < nTotTrait; i++) {
		for (j = 0; j < nIndPerGen; j++) {
			gsl_matrix_set(Liab, i, j, gsl_ran_gaussian(r, 1.0));
			if (nGen == 0)
				gsl_matrix_set(FstGenLiab, i, j, gsl_ran_gaussian(r, 1.0));
		}
	}

	if (nTotTrait > 1) {
		gsl_matrix_set_zero(tmpLiab);
		gsl_matrix_set_zero(tmpSigma);
		for (i = 0; i < nTotTrait; i++) {
			for (j = 0; j <= i; j++) {
				gsl_matrix_set(tmpSigma, i, j, gsl_matrix_get(Sigma, i, j));
			}
		}
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmpSigma, Liab, 0.0, tmpLiab);
		gsl_matrix_memcpy(Liab, tmpLiab);
		if (nGen == 0) {
			gsl_matrix_set_zero(tmpLiab);
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmpSigma, FstGenLiab, 0.0, tmpLiab);
			gsl_matrix_memcpy(FstGenLiab, tmpLiab);
		}
	}
}



void FirstGen() {
	/* generate first generation, no parents */
	int Gender;
	int i;
	double Age, HeritLiab[nMaxTrait], EnvLiab[nMaxTrait];
	double LiabDoI[nMaxTrait], Liab10Y[nMaxTrait]; 
	memset(HeritLiab, 0, sizeof(double) * nMaxTrait);
	memset(EnvLiab, 0, sizeof(double) * nMaxTrait);
	memset(LiabDoI, 0, sizeof(double) * nMaxTrait);
	memset(Liab10Y, 0, sizeof(double) * nMaxTrait);
	
	Age = gsl_rng_uniform(r); // need to multiple by the range to get age on date of interest // Var = 1/12
	Gender = (gsl_rng_uniform(r) < 0.5 ? 0 : 1); // 0 for female, 1 for male // Var = 0.25

	for (i = 0; i < nTotTrait; i ++) {
		HeritLiab[i] = gsl_matrix_get(FstGenLiab, i, nIndividual);
		HeritL[nGen][i][nIndividual] = HeritLiab[i];
		EnvLiab[i] = gsl_matrix_get(Liab, i, nIndividual);
		LiabDoI[i] = bAge[i]*Age + bSex[i]*Gender + bHerr[i]*HeritLiab[i] + bEnv[i]*EnvLiab[i];
		Liab10Y[i] = bAge[i]*(Age+10/(AgeRange[1]-AgeRange[0])) + bSex[i]*Gender + bHerr[i]*HeritLiab[i] + bEnv[i]*EnvLiab[i];
	}

	Trio[0] = 0;
	Trio[1] = 0;
	Trio[2] = (nGen+1) * nMaxIndPerGen + nIndividual; // Sample ID = (generation count+1)*nMaxIndPerGen + individual counter */
	Famlist[nGen][0][nIndividual] = Trio[0];
	Famlist[nGen][1][nIndividual] = Trio[1];
	Famlist[nGen][2][nIndividual] = Trio[2];
	Famlist[nGen][3][nIndividual] = Gender;
	WriteRes(Age, Gender, HeritLiab, EnvLiab, LiabDoI, Liab10Y);
}



/* check if Individual 1 and 2 are different in gender and unrelated within two generations 
0. Not same person lol; 1. no shared parent; 2. no shared grandparent;
return 1 if related, 0 if not*/
int CheckRelate(long int mo, long int fa) {
	long int tmpIdx;
	if ( (mo == fa) || Famlist[nGen-1][3][mo % nMaxIndPerGen] == Famlist[nGen-1][3][fa % nMaxIndPerGen]) { // same person or same gender
		return(1);
	}
	else if (nGen >= 2) {
		// Check if they share parents
		long int gmo1, gmo2, gfa1, gfa2;
		tmpIdx = (mo % nMaxIndPerGen); // Mother within generation index
		gmo1 = Famlist[nGen-1][0][tmpIdx];
		gfa1 = Famlist[nGen-1][1][tmpIdx];
		tmpIdx = (fa % nMaxIndPerGen); // Father within generation index
		gmo2 = Famlist[nGen-1][0][tmpIdx];
		gfa2 = Famlist[nGen-1][1][tmpIdx];
		if ((gmo1 == gmo2) || (gfa1 == gfa2)) {
			/* printf("parent1 = %ld, GrandMa1 = %ld, GrandPa1 = %ld\n", mo, gmo1, gfa1);
			printf("parent2 = %ld, GrandMa2 = %ld, GrandPa2 = %ld\n", fa, gmo2, gfa2);
			printf("Reject\n"); */
			return(1);
		}
		else if (nGen >= 3) {
			// Check if they share grandparent
			long int ggmo1, ggmo2, ggmo3, ggmo4, ggfa1, ggfa2, ggfa3, ggfa4;
			tmpIdx = (gmo1 % nMaxIndPerGen); // grandma1 within generation index
			ggmo1 = Famlist[nGen-2][0][tmpIdx];
			ggfa1 = Famlist[nGen-2][1][tmpIdx];
			tmpIdx = (gmo2 % nMaxIndPerGen); // grandma2 within generation index
			ggmo2 = Famlist[nGen-2][0][tmpIdx];
			ggfa2 = Famlist[nGen-2][1][tmpIdx];
			tmpIdx = (gfa1 % nMaxIndPerGen); // grandpa1 within generation index
			ggmo3 = Famlist[nGen-2][0][tmpIdx];
			ggfa3 = Famlist[nGen-2][1][tmpIdx];
			tmpIdx = (gfa2 % nMaxIndPerGen); // grandpa2 within generation index
			ggmo4 = Famlist[nGen-2][0][tmpIdx];
			ggfa4 = Famlist[nGen-2][1][tmpIdx];
			if ((ggmo1==ggmo2 || ggmo1==ggmo3 || ggmo1==ggmo4 || ggmo2==ggmo3 || ggmo2==ggmo4 || ggmo3==ggmo4) || 
				(ggfa1==ggfa2 || ggfa1==ggfa3 || ggfa1==ggfa4 || ggfa2==ggfa3 || ggfa2==ggfa4 || ggfa3==ggfa4)) {
				/* printf("GrandMa1 = %ld, gGrandMa1 = %ld, gGrandPa1 = %ld\n", gmo1, ggmo1, ggfa1);
				printf("GrandMa2 = %ld, gGrandMa2 = %ld, gGrandPa2 = %ld\n", gmo2, ggmo2, ggfa2);
				printf("GrandPa1 = %ld, gGrandMa3 = %ld, gGrandPa3 = %ld\n", gfa1, ggmo3, ggfa3);
				printf("GrandPa2 = %ld, gGrandMa4 = %ld, gGrandPa4 = %ld\n", gfa2, ggmo4, ggfa4);
				printf("Reject\n"); */
				return(1);
			}
		}
	}
	return(0);
}



void GetParent() {
	long int par1, par2;
	long int i;
	/* get parents */
	par1 = 0; 
	par2 = 0;
	// Get random parent from previous (nGen-1) generation
	while (CheckRelate(par1, par2)) {
		i = floor(gsl_rng_uniform(r) * nIndPerGen);
		par1 = Famlist[nGen-1][2][i];
		i = floor(gsl_rng_uniform(r) * nIndPerGen);
		par2 = Famlist[nGen-1][2][i];
	}
	Trio[Famlist[nGen-1][3][par1 % nMaxIndPerGen]] = par1;
	Trio[Famlist[nGen-1][3][par2 % nMaxIndPerGen]] = par2;
}



double FindParLiab(long int ID, int nTrait) {
	long int index = (ID % nMaxIndPerGen);
	double L = HeritL[nGen-1][nTrait][index];
	return(L);	
}



void GetLiab() {
	/* get liability for kid with individual 1 and 2 as parents */
	double MotherL, FatherL;
	int Gender;
	int i;
	double Age, HeritLiab[nMaxTrait], EnvLiab[nMaxTrait];
	double LiabDoI[nMaxTrait], Liab10Y[nMaxTrait]; // liability for individual by the day of interest and in 10 years prediction window (EndPtStatus and EndPtBefore)
	
	Age = gsl_rng_uniform(r); // need to multiple by the range to get age on date of interest
	Gender = (gsl_rng_uniform(r) < 0.5 ? 0 : 1); // 0 for female, 1 for male

	for (i = 0; i < nTotTrait; i++) {
		MotherL = FindParLiab(Trio[0], i);
		FatherL = FindParLiab(Trio[1], i);
		HeritLiab[i] = HerrCoeff[i]*(ParW[i][0]*MotherL + ParW[i][1]*FatherL);
		HeritL[nGen][i][nIndividual] = HeritLiab[i];
		EnvLiab[i] = gsl_matrix_get(Liab, i, nIndividual);
		LiabDoI[i] = bAge[i]*Age + bSex[i]*Gender + bHerr[i]*HeritLiab[i] + bEnv[i]*EnvLiab[i];
		Liab10Y[i] = bAge[i]*(Age+10/(AgeRange[1]-AgeRange[0])) + bSex[i]*Gender + bHerr[i]*HeritLiab[i] + bEnv[i]*EnvLiab[i];
	}
	Trio[2] = (nGen+1) * nMaxIndPerGen + nIndividual; // Sample ID = (generation count+1)*nMaxIndPerGen + individual counter */
	Famlist[nGen][0][nIndividual] = Trio[0];
	Famlist[nGen][1][nIndividual] = Trio[1];
	Famlist[nGen][2][nIndividual] = Trio[2];
	Famlist[nGen][3][nIndividual] = Gender;
	WriteRes(Age, Gender, HeritLiab, EnvLiab, LiabDoI, Liab10Y);
}



int main(int argc, char const *argv[]) {
	const gsl_rng_type * T;
	gsl_rng_env_setup();
    gsl_rng_default_seed = atoi(argv[2]);
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    ReadParam(argv[1]);
    GetBeta();
    MakeCovMat();

    memset(Trio, 0, sizeof(long int) * 3);
    nGen = 0;
    LiabGenerator();
    long int i; // individual counter
    for (nIndividual = 0; nIndividual < nIndPerGen; nIndividual++) {
    	memset(Trio, 0, sizeof(long int) * 3);
    	FirstGen();
    }

    for (nGen = 1; nGen < nTotGen; nGen++) {
    	LiabGenerator();
    	for (nIndividual = 0; nIndividual < nIndPerGen; nIndividual++) {
    		memset(Trio, 0, sizeof(long int) * 3);
    		GetParent();
    		GetLiab();
    	}
    }
    printf("Pedigree generation done! Enjoy your families :D\n");
    return(0);
}


