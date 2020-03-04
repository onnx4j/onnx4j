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
package org.onnx4j.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.onnx4j.Model;
import org.onnx4j.NamedOnnxObject;
import org.onnx4j.model.graph.Constant;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.exchanges.GraphInput;
import org.onnx4j.model.graph.exchanges.GraphOutput;
import org.onnx4j.prototypes.OnnxProto3.GraphProto;
import org.onnx4j.prototypes.OnnxProto3.NodeProto;
import org.onnx4j.prototypes.OnnxProto3.TensorProto;
import org.onnx4j.prototypes.OnnxProto3.ValueInfoProto;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.graph.GraphBuilder;
import com.google.common.graph.ImmutableGraph.Builder;

public class Graph extends NamedOnnxObject {

	private static Logger logger = LoggerFactory.getLogger(Graph.class);

	private Model model;
	private com.google.common.graph.Graph<Node> dag;
	private Constant[] constants;
	private GraphInput[] inputs;
	private GraphOutput[] outputs;

	public Graph(Model model, GraphProto graphProto) {
		super(graphProto.getName(), graphProto.getDocString());
		
		this.model = model;

		Map<String, Node> nodeMapByOutName = new HashMap<String, Node>();
		Map<String, Collection<Node>> nodesMapByInName = new HashMap<String, Collection<Node>>();

		//
		// ONNX定义中的node，一般指代ONNX4J中的OperationNode，应存在入度与出度（若为输出节点，则不存在）
		//
		for (NodeProto nodeProto : graphProto.getNodeList()) {
			Node node = new Node(this.model, nodeProto, this.model.getTensorOptions());

			//
			// 保存输出名称引用，为下阶段计算依赖关系准备
			//
			for (String outputName : nodeProto.getOutputList()) {
				nodeMapByOutName.put(outputName, node);
			}

			//
			// 保存输入名称引用，为下阶段计算依赖关系准备
			// 同一个输入名称，可能对应多个Node
			//
			for (String inputName : nodeProto.getInputList()) {
				Collection<Node> nodes = nodesMapByInName.get(inputName);
				if (nodes == null) {
					nodes = new ArrayList<Node>();
					nodesMapByInName.put(inputName, nodes);
				}
				nodes.add(node);
			}
		}

		this.constants = this.initConstants(graphProto);
		assert this.constants != null;

		this.inputs = this.initInputs(graphProto);
		assert this.inputs != null && this.inputs.length > 0;

		this.dag = this.buildDAG(graphProto, nodeMapByOutName, nodesMapByInName);
		assert this.dag != null;
		logger.debug("The definition of graph \"{}\": \"{}\"", super.name, this.dag);

		this.outputs = this.initOutputs(graphProto, nodeMapByOutName);
		assert this.outputs != null && this.outputs.length > 0;
	}

	public GraphInput[] getInputs() {
		return this.inputs;
	}

	public GraphInput getInputs(String inputName) {
		for (GraphInput graphInput : this.inputs) {
			if (graphInput.getName().equalsIgnoreCase(inputName))
				return graphInput;
		}

		return null;
	}

	public GraphOutput[] getOutputs() {
		return this.outputs;
	}

	public GraphOutput getOutput(String outputName) {
		for (GraphOutput graphOutput : this.outputs) {
			if (graphOutput.getName().equalsIgnoreCase(outputName))
				return graphOutput;
		}

		return null;
	}

	public Constant[] getConstants() {
		return this.constants;
	}

	/**
	 * 返回指定节点的前辈节点集合
	 * 
	 * @param node
	 * @return
	 */
	public Set<Node> predecessors(Node node) {
		return this.dag.predecessors(node);
	}

	/**
	 * 返回指定节点的集成人节点集合
	 * 
	 * @param node
	 * @return
	 */
	public Set<Node> successors(Node node) {
		return this.dag.successors(node);
	}

	public Set<Node> getNodes() {
		return this.dag.nodes();
	}

	public Node getNode(String nodeName) {
		for (Node node : this.dag.nodes()) {
			if (node.getName().equalsIgnoreCase(nodeName))
				return node;
		}

		return null;
	}

	private Constant[] initConstants(GraphProto graph) {
		//
		// 作为输入常量，不存在入度，即不存在依赖节点
		// 区别与输入节点，此节点在执行时不需要用户喂入(feed)运行时数据，由网络构建时定义好数值
		//
		List<TensorProto> initializerList = graph.getInitializerList();
		Constant[] contants = new Constant[initializerList.size()];
		for (int n = 0; n < initializerList.size(); n++) {
			TensorProto initializer = initializerList.get(n);

			//
			// 保存输出名称引用，为下阶段计算依赖关系准备
			//
			contants[n] = new Constant(this.model, initializer);
		}

		return contants;
	}

	private GraphInput[] initInputs(GraphProto graph) {
		//
		// 作为网络输入，不存在入度，即不存在依赖节点
		//
		List<ValueInfoProto> inputList = graph.getInputList();
		GraphInput[] exchanges = new GraphInput[inputList.size()];
		for (int n = 0; n < inputList.size(); n++) {
			ValueInfoProto valueInfoProto = inputList.get(n);

			//
			// 保存输出名称引用，为下阶段计算依赖关系准备
			//
			exchanges[n] = new GraphInput(valueInfoProto);

			logger.debug("Input named \"{}\" in Graph \"{}\"", exchanges[n].getName(), super.getName());
		}

		return exchanges;
	}

	private GraphOutput[] initOutputs(GraphProto graph, Map<String, Node> nodeMapByOutName) {
		//
		// 作为网络输入，不存在入度，即不存在依赖节点
		//
		List<ValueInfoProto> outputList = graph.getOutputList();
		GraphOutput[] exchanges = new GraphOutput[outputList.size()];
		for (int n = 0; n < outputList.size(); n++) {
			ValueInfoProto valueInfoProto = outputList.get(n);

			//
			// 保存输出名称引用，为下阶段计算依赖关系准备
			//
			Node node = nodeMapByOutName.get(valueInfoProto.getName());
			exchanges[n] = new GraphOutput(node, valueInfoProto);

			logger.debug("Output named \"{}.{}\" in Graph \"{}\"", exchanges[n].getNode().getName(),
					exchanges[n].getName(), super.getName());
		}

		return exchanges;
	}

	private com.google.common.graph.Graph<Node> buildDAG(GraphProto graphProto, Map<String, Node> nodeMapByOutName,
			Map<String, Collection<Node>> nodesMapByInName) {
		Builder<Node> builder = GraphBuilder.directed().allowsSelfLoops(false).<Node>immutable();

		for (Entry<String, Node> entrySet : nodeMapByOutName.entrySet()) {
			String outputName = entrySet.getKey();
			Node outputNode = entrySet.getValue();
			Collection<Node> inNodes = nodesMapByInName.get(outputName);
			if (inNodes != null) {
				for (Node inNode : inNodes) {
					builder.putEdge(outputNode, inNode);
				}
			}
		}

		return builder.build();
	}

}