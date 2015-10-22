package ml.learn.tree;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import edu.stanford.nlp.parser.metrics.Evalb;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.DependencyTreeTransformer;
import edu.stanford.nlp.trees.PennTreeReader;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import ml.learn.object.BinaryTree;

public class Main {
	public static void main(String[] args) throws IOException{
		List<BinaryTree> trainingData = getTrainingData("ptb.dev");
		trainingData = trainingData.subList(0, 40);
//		List<BinaryTree> trainingData = getTrainingData(null);
//		System.out.println("Converted");
//		System.out.println(trainingData.get(0));
		PCFG pcfg = new PCFG(trainingData);
		EMAlgo emAlgo = new EMAlgo(pcfg);

		double[] startingPoint = new double[pcfg.numParams()];
		System.out.println("Starting point:");
		for(int i=0; i<startingPoint.length; i++){
			startingPoint[i] = 1.0/startingPoint.length;
			System.out.printf("%.3f ", startingPoint[i]);
		};
		System.out.println();
		
		double[] bestParams = emAlgo.getBestParams(startingPoint);
		System.out.println("Best params:");
		for(double bestParam: bestParams){
			System.out.printf("%.3f ", bestParam);
		}
		System.out.println();
		
//		List<BinaryTree> testData = getTestData(null);
		List<BinaryTree> testData = getTestData("ptb.dev");
		testData = testData.subList(0, 10);
		Evalb metric = new Evalb("Evaluation", true);
		for(BinaryTree tree: testData){
			if(tree.terminals == null) tree.fillTerminals();
			BinaryTree predicted = pcfg.predict(tree.terminals, bestParams);
			Tree stanfordActual = toStanfordTree(tree);
			Tree stanfordPredicted = toStanfordTree(predicted);
			metric.evaluate(stanfordPredicted, stanfordActual, null);
			System.out.println("Actual");
			System.out.println(stanfordActual.pennString());
			System.out.println("Predicted");
			System.out.println(stanfordPredicted.pennString());
		}
		metric.display(true);
	}
	
	/**
	 * Convert our {@link BinaryTree} into Stanford {@link Tree}
	 * @param tree
	 * @return
	 */
	private static Tree toStanfordTree(BinaryTree tree){
		try{
			PennTreeReader reader = new PennTreeReader(new StringReader(tree.toString()));
			Tree result = reader.readTree();
			reader.close();
			return result;
		} catch (Exception e){
			return null;
		}
	}
	
	/**
	 * Return training data.
	 * If fileName is null, a small artificial data is returned.
	 * @param fileName
	 * @return
	 * @throws IOException
	 */
	private static List<BinaryTree> getTrainingData(String fileName) throws IOException{
		List<BinaryTree> result;
		if(fileName == null){
			result = new ArrayList<BinaryTree>();
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NN duck))(VP (VBZ is)(VBG swimming)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NNS cats))(VP (VBP are)(VBG running)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT A)(NN dog))(VP (VBD was)(VBG swimming)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT A)(NN duck))(VP (VBZ is)(VBG running)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NN dog))(VP (VBD is)(VBG swimming)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NN dog))(VP (VBZ ducks)(RB hastily)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NNS cats))(VP (VBP duck)(RB hastily)))"));
		} else {
			result = readPTB(fileName);
		}
		return result;
	}

	
	/**
	 * Return test data.
	 * If fileName is null, a small artificial data is returned.
	 * @param fileName
	 * @return
	 * @throws IOException
	 */
	private static List<BinaryTree> getTestData(String fileName) throws IOException{
		List<BinaryTree> result;
		if(fileName == null){
			result = new ArrayList<BinaryTree>();
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NN dog))(VP (VBZ is)(VBG swimming)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NNS cats))(VP (VBP are)(VBG running)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT A)(NN duck))(VP (VBD was)(VBG swimming)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT A)(NN dog))(VP (VBZ is)(VBG running)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NN dog))(VP (VBD is)(VBG swimming)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NN dog))(VP (VBZ ducks)(RB hastily)))"));
			result.add(BinaryTree.parse("(ROOT (NP (DT The)(NNS cats))(VP (VBP duck)(RB hastily)))"));
		} else {
			result = readPTB(fileName);
		}
		return result;
	}
	
	/**
	 * Read the trees from a file containing list of serialized tree in Penn Treebank bracketed format.
	 * Will convert the trees into binary trees.
	 * @param fileName
	 * @return
	 * @throws IOException
	 */
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
	
	/**
	 * Read Stanford Tree from the given serialized tree in Penn Treebank bracketed format
	 * @param input
	 * @return
	 */
	private static Tree readStanfordTree(String input){
		Tree tree = null;
		try{
			PennTreeReader reader = new PennTreeReader(new StringReader(input));
			tree = reader.readTree();
//			System.out.println("Original");
//			System.out.println(tree.pennString());
			DependencyTreeTransformer depTreeTransformer = new DependencyTreeTransformer();
			tree = depTreeTransformer.transformTree(tree);
//			System.out.println("Transformed");
//			System.out.println(tree.pennString());
			TreeBinarizer binarizer = TreeBinarizer.simpleTreeBinarizer(new CollinsHeadFinder(), new PennTreebankLanguagePack());
			tree = binarizer.transformTree(tree);
//			System.out.println("Binarized");
//			System.out.println(tree.pennString());
			reader.close();
		} catch (IOException e){}
		return tree;
	}
}
