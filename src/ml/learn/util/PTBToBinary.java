package ml.learn.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.DependencyTreeTransformer;
import edu.stanford.nlp.trees.PennTreeReader;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import ml.learn.object.BinaryTree;

public class PTBToBinary {
	
	public static void main(String[] args) throws FileNotFoundException {
		String input = args[0];
		String output = args[1];
		System.out.println("Converting PTB from \""+input+"\" to \""+output+"\"");
		List<BinaryTree> trees = readPTB(input);
		try (PrintWriter writer = new PrintWriter(output)){
			for(BinaryTree tree: trees){
				writer.println(toStanfordTree(tree).toString().replace(") ", ")"));
			}
		}
		System.out.println("Conversion done");
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
	 * Read the trees from a file containing list of serialized tree in Penn Treebank bracketed format.
	 * Will convert the trees into binary trees.
	 * @param fileName
	 * @return
	 * @throws IOException
	 */
	private static List<BinaryTree> readPTB(String fileName) throws FileNotFoundException {
		List<BinaryTree> result = new ArrayList<BinaryTree>();
		Scanner sc = new Scanner(new File(fileName), "UTF-8");
		while(sc.hasNextLine()){
			String line = sc.nextLine();
			Tree stanfordTree = readStanfordTree(line);
			BinaryTree tree = BinaryTree.fromStanfordTree(stanfordTree);
			result.add(tree);
		}
		sc.close();
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
