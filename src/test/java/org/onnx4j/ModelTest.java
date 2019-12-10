package org.onnx4j;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URLDecoder;
import java.util.Set;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import org.onnx4j.model.Graph;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.exchanges.GraphOutput;

import com.google.common.graph.GraphBuilder;
import com.google.common.graph.ImmutableGraph;

/**
 * Unit test for simple App.
 */
public class ModelTest extends TestCase {
	/**
	 * Create the test case
	 *
	 * @param testName
	 *            name of the test case
	 */
	public ModelTest(String testName) {
		super(testName);
	}

	/**
	 * @return the suite of tests being tested
	 */
	public static Test suite() {
		return new TestSuite(ModelTest.class);
	}

	/**
	 * Rigourous Test :-)
	 * 
	 * @throws IOException 
	 * @throws FileNotFoundException 
	 */
	public void testLoad() throws FileNotFoundException, IOException {
		String modelPath = URLDecoder.decode(ModelTest.class.getResource("/simple_tf.onnx").getFile(), "utf-8");
		assertNotNull(modelPath);

		Model model = new Model(modelPath);
		assertNotNull(model);
		
		Graph g = model.getGraph();
		GraphOutput[] outputs = g.getOutputs();
		for (GraphOutput output : outputs) {
			this.handle(g, output.getNode());
		}
	}
	
	private void handle(Graph g, Node node) {
		Set<Node> set = g.predecessors(node);
		for (Node predecessorNode : set) {
			this.handle(g, predecessorNode);
		}
		System.out.println(node.getName());
	}
	
	public void testGuavaGraph() {
		ImmutableGraph<Integer> graph =
			    GraphBuilder.directed()
			        .<Integer>immutable()
			        .addNode(1)
			        .putEdge(2, 3) // also adds nodes 2 and 3 if not already present
			        .putEdge(2, 3) // no effect; Graph does not support parallel edges
			        .putEdge(3, 4)
			        .build();
		Set<Integer> successorsOfTwo = graph.successors(2); // returns {3}
		assertEquals(1, successorsOfTwo.size());
		assertTrue(successorsOfTwo.contains(3));
		Set<Integer> successorsOfFour = graph.predecessors(4); // returns {3}
		assertEquals(1, successorsOfFour.size());
		assertTrue(successorsOfFour.contains(3));
	}
	
}
