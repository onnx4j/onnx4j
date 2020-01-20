/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j;

import java.net.URLDecoder;
import java.util.Set;

import org.onnx4j.model.Graph;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.exchanges.GraphOutput;

import com.google.common.graph.GraphBuilder;
import com.google.common.graph.ImmutableGraph;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for class of Model.
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
	 * @throws Exception 
	 */
	public void testLoad() throws Exception {
		String modelPath = URLDecoder.decode(ModelTest.class.getResource("/simple_tf.onnx").getFile(), "utf-8");
		assertNotNull(modelPath);

		try (Model model = new Model(modelPath)) {
			assertNotNull(model);
			
			Graph g = model.getGraph();
			GraphOutput[] outputs = g.getOutputs();
			for (GraphOutput output : outputs) {
				this.handle(g, output.getNode());
			}
			
			model.close();
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