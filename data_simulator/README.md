# Synthetic data simulator

Code to generate synthetic data used for this project.

Required library: [gsl](https://www.gnu.org/software/gsl/)

## Compilation

Download `SimPedPheno_V1.1.c` file and run
<pre>
gcc SimPedPheno_V1.1.c -o <em>PhenoPedSim</em> -L<em>gsl_lib_directory</em> -I<em>gsl_include_directory</em> -lm -lgsl -fPIC -lcblas -lblas
</pre>
It should creat an exicutable named <em>PhenoPedSim</em> in the current directory.

## Basic usage

Run
<pre>
./PhenoPedSim <em>ParFile</em>  <em>seed</em>
</pre>
to generate synthetic pedigree data and phenotypeic liability, where <em>seed</em> is a numerial seed for randomization, and <em>ParFile</em> is a plain text parameter file that could look like below:
<pre>
Prevalence  0.3
AgeLower  18
AgeUpper  70
MaternalWeight  0.5
AgeEffect 0.1
SexEffect 0.1
Herritable  0.2
nGeneration 3
nIndividual 3000
OutPrefix TestOut
</pre>

`Prevalence` is the approximated population level prevalence of the phenotype you would like to create.

`AgeLower` and `AgeUpper` are the lower and upper bound of synthetic individual age on the date of interest.

`MaternalWeight` is a \[0,1\] value noting the countribution of familial risk from one's mother. The familial risk the father side will therefore be _1-MaternalWeight_

`AgeEffect` and `SexEffect` are \[0,1\] values noting the proportion of disease liability variance explained by one's age and gender respectively; and `Herritable` is a \[0,1\] value noting the proportion of disease liability variance explained by familial risk. The rest of disease liability will be individual noise. Therefore, we expect an input with _AgeEffect_ + _SexEffect_ + _Herritable_ <= 1.

`nGeneration` is an integer indicating total number of generations you would like to simulate. Current maximum is 10. You can increase it by changing the number in line 14 of the _SimPedPheno_V1.1.c_ file `#define nMaxGen 10` and recompile.

`nIndividual` is an integer indicating total number of individuals you would like to simulate **for each generation**. Current maximum is 100000. You can increase it by changing the number in line 13 of the _SimPedPheno_V1.1.c_ file `#define nMaxIndPerGen 100000` and recompile.

`OutPrefix` is your prefered location and prefix for the output.

* You can also simulate multiple correlated phenotypes (maximum 10, can be changed on line 15 of _SimPedPheno_V1.1.c_) if needed. In this case, `Prevalence`, `MaternalWeight`, `AgeEffect`, `SexEffect`, `Herritable` should be comma seperated values for each phenotye of interest, and one additional parameter `TraitCorrelation` indicating the correlation of **familal and enviromental (noise) risk** between each phenotype and the first one. So the first element of `TraitCorrelation` should always be 1, since it is the correlation between the first trait and itself. An example parameter file for multitrait simulation can be found below:

<pre>
Prevalence 0.3,0.1
AgeLower 18
AgeUpper 70
MaternalWeight 0.8,0.5
AgeEffect 0.1,0.2
SexEffect 0.1,0.1
Herritable 0.2,0.4
TraitCorrelation 1,0.3
nGeneration 3
nIndividual 3000
OutPrefix TestOut
</pre>


## Output

Running the command above will create an output file specificed by the `OutPrefix` parameterm, that looks like below:
<pre>
Mother	Father	Offspring	AgeDoI	AgeDoILiab-1	Age10YLiab-1	Gender	GenderLiab-1	HeritLiab-1	EnvLiab-1	LiabDoI-1	Liab10Y-1	EndPtDoI-1	EndPt10Y-1
0	0	100000	55.06	0.7807	0.9914	1	0.6325	-0.2532	0.8411	2.0011	2.2118	1	1
0	0	100001	43.19	0.5308	0.7414	1	0.6325	0.7780	0.5653	2.5065	2.7172	1	1
0	0	100002	60.29	0.8910	1.1016	1	0.6325	0.6358	-0.0568	2.1024	2.3131	1	1
0	0	100003	32.44	0.3041	0.5148	1	0.6325	0.3378	-0.0422	1.2322	1.4428	0	1
0	0	100004	40.26	0.4689	0.6796	0	0.0000	-0.6701	-0.8846	-1.0858	-0.8752	0	0
0	0	100005	60.28	0.8907	1.1014	0	0.0000	-0.4036	0.9296	1.4167	1.6274	1	1
0	0	100006	62.21	0.9312	1.1419	0	0.0000	0.0703	-1.2089	-0.2074	0.0033	0	0
0	0	100007	40.07	0.4650	0.6757	1	0.6325	0.3380	1.1888	2.6242	2.8349	1	1
0	0	100008	41.86	0.5027	0.7134	1	0.6325	0.8135	-0.1893	1.7594	1.9701	1	1
</pre>

`Mother` and `Father` denote the parents of each synthetic individual. 0 indicates the individual is from the ancestry generation (generation 1). `Offspring` is the individual that current row of information corresponds to. First digit of the individual ID indicates the generation that one is from. e.g. 100008 indicates this is the 9-th individual (individual counter starts from 0, so 8+1) generated for the 1st generation; and 300240 would be the 241-th for the 3rd generation.

`AgeDoI` is `Offspring`'s age at date of interest. It should be within the rage you specified with `AgeLower` and `AgeUpper`. 
* Note that in the output of this simulation, we may see offspring's age > parents' age. Therefore `AgeDoI` can also be conceptually interprated as "the maximum age one could reach by the date of interest".

`AgeDoILiab` and `Age10YLiab` are the liability contributed by risk from age on date of interest and on 10 years after date of interest respectively. The "10 years risk" was designed for our prediction task described in the manuscrupt. 

`Gender` is one's synthetic gender (0 is female and 1 is male) and `GenderLiab` is the amount of liability contributed by gender specific risk.

`HeritLiab` and `EnvLiab` are libility from familial and enveriomental risk.

`LiabDoI` is total phenotypic liability for the synthetic individual on the date of interest, and `Liab10Y` is his phenotypic liability after 10 years.

`EndPtDoI` and `EndPt10Y` are their binary phenotypic status derived given the specified `Prevalence`.

* The `-1` labels in the output header indicates the libilities for phenotype 1. When simulating multiple phenoytpes, the output will look like below:
<pre>
Mother	Father	Offspring	AgeDoI	AgeDoILiab-1,AgeDoILiab-2	Age10YLiab-1,Age10YLiab-2	Gender	GenderLiab-1,GenderLiab-2	HeritLiab-1,HeritLiab-2	EnvLiab-1,EnvLiab-2	LiabDoI-1,LiabDoI-2	Liab10Y-1,Liab10Y-2	EndPtDoI-1,EndPtDoI-2	EndPt10Y-1,EndPt10Y-2
0	0	100000	23.05	0.1064,0.1505	0.3171,0.4484	1	0.6325,0.6325	-0.2532,0.2176	0.8411,1.4910	1.3268,2.4915	1.5374,2.7894	0,1	1,1
0	0	100001	54.46	0.7680,1.0861	0.9787,1.3840	1	0.6325,0.6325	0.7780,0.1213	0.5653,-0.0683	2.7438,1.7715	2.9544,2.0694	1,0	1,0
0	0	100002	47.74	0.6265,0.8860	0.8372,1.1840	0	0.0000,0.0000	0.6358,0.7276	-0.0568,-0.2255	1.2055,1.3881	1.4162,1.6860	0,0	1,0
0	0	100003	64.72	0.9843,1.3920	1.1949,1.6899	0	0.0000,0.0000	0.3378,-0.9284	-0.0422,1.2043	1.2799,1.6679	1.4906,1.9658	0,0	1,0
0	0	100004	65.78	1.0066,1.4235	1.2172,1.7214	1	0.6325,0.6325	-0.6701,-0.5412	-0.8846,1.1172	0.0843,2.6319	0.2949,2.9299	0,1	0,1
0	0	100005	48.55	0.6435,0.9100	0.8542,1.2080	1	0.6325,0.6325	-0.4036,-0.9815	0.9296,0.3922	1.8020,0.9532	2.0126,1.2512	1,0	1,0
0	0	100006	56.65	0.8141,1.1514	1.0248,1.4493	1	0.6325,0.6325	0.0703,-0.0445	-1.2089,-0.9044	0.3080,0.8350	0.5186,1.1329	0,0	0,0
0	0	100007	19.48	0.0311,0.0440	0.2418,0.3419	1	0.6325,0.6325	0.3380,1.5162	1.1888,0.2620	2.1903,2.4546	2.4010,2.7525	1,1	1,1
0	0	100008	18.44	0.0094,0.0132	0.2200,0.3112	0	0.0000,0.0000	0.8135,-1.0937	-0.1893,0.3452	0.6336,-0.7353	0.8442,-0.4374	0,0	0,0
</pre>
with outcome of each phenotype seperated by comma.






