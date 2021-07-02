import java.util.*;

public class Manchot{
    public double esperence;
    public double variance; 
    private Random gaussien;

    public Manchot(double esperence, double variance){
        this.esperence = esperence;
        this.variance = variance;
        gaussien = new Random();
    }

    public double tirerBras(){
        return gaussien.nextGaussian()*variance +esperence;
    }

    public static double rechercheAleatoire(int nbIterations, Manchot[] manchot){
        double somme = 0;
        Random r = new Random();

        for(int i=0; i<nbIterations; i++){
            somme += manchot[r.nextInt(manchot.length)].tirerBras();
        }
        return somme;
    }

    public static double rechercheGloutonne(int nbIterations, Manchot[] manchot){
        double somme= 0;
        double  tmp1 = 0;
        int indexBest = 0;
        int i = 0;

        for(i=0; i < manchot.length-1; i++){
            tmp1 = manchot[i].tirerBras(); 
            if(tmp1 <  manchot[i+1].tirerBras()){
                tmp1 = manchot[i+1].tirerBras();
                indexBest = i+1;                        
            }
            else 
                indexBest = i;
        }

        for(int j=indexBest; j<nbIterations-manchot.length; j++){
            somme += manchot[indexBest].tirerBras();
        }
        return somme; 
    }

    public static double rechercheUCB(int nbIterations, Manchot[] manchot){
        int[] nbTiragesBras = new int[manchot.length];
        double[] sommeBras = new double[manchot.length];
        int i = 0;
        double k = 2;

        for(i = 0; i<manchot.length; i++){
            nbTiragesBras[i] += 1;
            sommeBras[i] += manchot[i].tirerBras();
        }
        for(int j=i; j<nbIterations-manchot.length; j++){
            double[] ucb = new double[manchot.length]; 

            for(int p=0; p<manchot.length; p++)
                ucb[p] = sommeBras[p] + k * Math.sqrt( Math.log(nbIterations) / (nbTiragesBras[p]) );

            double bestucb = ucb[0];
            int bestIndex = 0;
            for(int o=0; o<manchot.length; o++){
                if(bestucb < ucb[o]){
                    bestucb = ucb[o];
                    bestIndex = o;
                }
            }
            sommeBras[bestIndex] += manchot[bestIndex].tirerBras();
            nbTiragesBras[bestIndex] += 1;
        }
        double total = 0;
        for(double gain:sommeBras)
            total += gain;
        return total;
    }
    

    public static Manchot[] creerManchots(int nb){
        Manchot[] manchots = new Manchot[nb];
        for(int i=0; i<nb; i++){
            double rvariance = Math.random() * (10+1);
            double resperence = Math.random() * (10 - (-10) + 1);
            Manchot manchot = new Manchot(rvariance, resperence);
            manchots[i] = manchot;
        }
        return manchots;
    }

    public static void main(String[] args){
        Manchot[] manchot = creerManchots(15);
        System.out.println("random :" + rechercheAleatoire(15000, manchot));
        System.out.println("glouton :" + rechercheGloutonne(15000, manchot));
        System.out.println("UCB :" + rechercheUCB(15000, manchot));
    }
}


/*
 ============REPONSES AUX QUESTIONS============= 
1) Si K devient très grand, UCB se rapproche de l'algorithme random, et si K est petit, on se rapproche d'un algorithme glouton. 
2) Lorsque l'on prend une plus petit variance, l'algorithme glouton sera, dans la plupart des cas, meilleur que UCB. (ici : [0;5])
3) Lorsque l'on prend une grand variance, UCB sera quasiment toujours meilleur que l'algorithme glouton. (ici [0;100])
*/