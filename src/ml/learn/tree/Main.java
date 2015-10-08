package ml.learn.tree;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.DependencyTreeTransformer;
import edu.stanford.nlp.trees.PennTreeReader;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import ml.learn.object.BinaryTree;

public class Main {
	public static void main(String[] args) throws IOException{
		Random random = new Random(0);
		
		List<BinaryTree> trainingData = getTrainingData("ptb.dev");
//		List<BinaryTree> trainingData = getTrainingData(null);
		trainingData = trainingData.subList(0, 5);
		EMCompatibleFunction pcfg = new PCFG(trainingData);
		EMAlgo emAlgo = new EMAlgo(pcfg);

		double[] startingPoint = new double[pcfg.numParams()];
		System.out.println("Starting point:");
		for(int i=0; i<startingPoint.length; i++){
			startingPoint[i] = random.nextGaussian();
			System.out.printf("%.3f ", startingPoint[i]);
		};
		System.out.println();
		
		System.out.println("Best params:");
		double[] bestParams = emAlgo.getBestParams(startingPoint);
		for(double bestParam: bestParams){
			System.out.printf("%.3f ", bestParam);
		}
		System.out.println();
	}
	
	private static List<BinaryTree> getTrainingData(String fileName) throws IOException{
		List<BinaryTree> result;
		if(fileName == null){
			result = new ArrayList<BinaryTree>();
			result.add(BinaryTree.parse("(S (NP (DT The)(NN duck))(VP (VBZ is)(VBG swimming)))"));
			result.add(BinaryTree.parse("(S (NP (DT The)(NNS cats))(VP (VBP are)(VBG running)))"));
			result.add(BinaryTree.parse("(S (NP (DT A)(NN dog))(VP (VBD was)(VBG swimming)))"));
			result.add(BinaryTree.parse("(S (NP (DT A)(NN duck))(VP (VBZ is)(VBG running)))"));
			result.add(BinaryTree.parse("(S (NP (DT The)(NN dog))(VP (VBD is)(VBG swimming)))"));
		} else {
			result = readPTB(fileName);
		}
		return result;
	}
	
	private static List<BinaryTree> readPTB(String fileName) throws IOException{
		List<BinaryTree> result = new ArrayList<BinaryTree>();
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
		while(reader.ready()){
			String line = reader.readLine();
			Tree stanfordTree = readStanfordTree(line);
			BinaryTree tree = BinaryTree.fromStanfordTree(stanfordTree);
			result.add(tree);
		}
		reader.close();
		return result;
	}
	
	private static Tree readStanfordTree(String input){
		Tree tree = null;
		try{
			PennTreeReader reader = new PennTreeReader(new StringReader(input));
			tree = reader.readTree();
			DependencyTreeTransformer depTreeTransformer = new DependencyTreeTransformer();
			tree = depTreeTransformer.transformTree(tree);
			TreeBinarizer binarizer = TreeBinarizer.simpleTreeBinarizer(new CollinsHeadFinder(), new PennTreebankLanguagePack());
			tree = binarizer.transformTree(tree);
			reader.close();
		} catch (IOException e){}
		return tree;
	}
}
