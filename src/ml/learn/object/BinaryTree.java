package ml.learn.object;

import java.security.InvalidParameterException;
import java.text.CharacterIterator;
import java.text.StringCharacterIterator;

import edu.stanford.nlp.trees.Tree;

/**
 * Represents a binary tree (a tree with exactly two children)
 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
 *
 */
public class BinaryTree {
	public BinaryTree left;
	public BinaryTree right;
	public TaggedWord value;
	
	public String[] terminals;
	
	public BinaryTree(){
		
	}
	
	public BinaryTree(TaggedWord value){
		this.value = value;
	}
	
	public void fillTerminals(){
		if(isTerminal()){
			terminals = new String[]{value.word()};
		} else {
			left.fillTerminals();
			right.fillTerminals();
			String[] leftTerminals = left.terminals;
			String[] rightTerminals = right.terminals;
			int leftLen = leftTerminals.length;
			int rightLen = rightTerminals.length;
			int thisLen = leftLen + rightLen;
			terminals = new String[thisLen];
			for(int i=0; i<leftLen; i++){
				terminals[i] = leftTerminals[i];
			}
			for(int i=0; i<rightLen; i++){
				terminals[leftLen+i] = rightTerminals[i];
			}
		}
	}
	
	public static BinaryTree fromStanfordTree(Tree tree){
		BinaryTree result = new BinaryTree();
		String label = tree.value();
		int dashIdx = label.indexOf("-");
		if(dashIdx > -1){
			label = label.substring(0, dashIdx);
		}
		if(label.startsWith("@")){
			label = label.substring(1);
		}
		if(tree.isPhrasal() && !tree.isPreTerminal()){
			if(tree.numChildren() == 2){
				result.left = fromStanfordTree(tree.firstChild());
				result.right = fromStanfordTree(tree.lastChild());
				result.value = new TaggedWord("", Tag.get(label));
			} else {
				result = fromStanfordTree(tree.firstChild());
				result.value = new TaggedWord(result.value.word(), Tag.get(label));
			}
		} else {
			result.value = new TaggedWord(tree.firstChild().value(), Tag.get(label));
		}
		return result;
	}
	
	public static BinaryTree parse(String bracketedTree){
		CharacterIterator iterator = new StringCharacterIterator(bracketedTree);
		return parse(iterator);
	}
	
	public static BinaryTree parse(CharacterIterator input) throws InvalidParameterException {
		BinaryTree result = new BinaryTree();
		StringBuilder read = new StringBuilder();
		try{
			char c;
			String label = "";
			while((c=next(read, input)) != ' '){
				label += c;
			}
			
			// Take only the base label (e.g., NP-SBJ-1 into NP)
			int dashIndex = label.indexOf("-");
			if(dashIndex != -1){
				label = label.substring(0, dashIndex);
			}
			if((c=next(read, input)) == '('){ // Another tree
				result.left = parse(input);
				result.right = parse(input);
				result.value = new TaggedWord("", Tag.get(label));
			} else { // Word
				String word = ""+c;
				while((c=next(read, input)) != ')'){
					word += c;
				}
				result.value = new TaggedWord(word, Tag.get(label));
			}
			input.next();
		} catch (Exception e){
			System.out.println(getReadTree(read, input));
			throw e;
		}
		return result;
	}
	
	/**
	 * Get the next character in the iterator, saving it to the given StringBuilder while also returning it
	 * @param read
	 * @param rest
	 * @return
	 */
	private static char next(StringBuilder read, CharacterIterator rest){
		char c = rest.next();
		if(c != CharacterIterator.DONE) read.append(c);
		return c;
	}
	
	/**
	 * Return the tree given to the parser
	 * @param read
	 * @param rest
	 * @return
	 */
	private static String getReadTree(StringBuilder read, CharacterIterator rest){
		char c;
		while((c=rest.next()) != CharacterIterator.DONE) read.append(c);
		return read.toString();
	}
	
	/**
	 * Whether this tree node is a terminal
	 * @return
	 */
	public boolean isTerminal(){
		return left == null && right == null;
	}
	
	public String toString(){
		return toString(0);
	}
	
	public String toString(int level){
		StringBuilder builder = new StringBuilder();
		if(left == null){
			for(int i=0; i<level; i++) builder.append("  ");
			builder.append("("+value.tag()+" "+value.word()+")");
		} else {
			for(int i=0; i<level; i++) builder.append("  ");
			builder.append("(");
			builder.append(value.tag());
			builder.append("\n");
			builder.append(left.toString(level+1));
			builder.append("\n");
			builder.append(right.toString(level+1));
			builder.append(")");
		}
		return builder.toString();
	}
}
