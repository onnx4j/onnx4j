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
package org.onnx4j.model.graph;

import java.util.Arrays;

import org.onnx4j.Model;
import org.onnx4j.NamedOnnxObject;
import org.onnx4j.Tensor;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.prototypes.OnnxProto3.NodeProto;

public final class Node extends NamedOnnxObject {

	protected Model model;
	protected String domain;
	protected String opType;
	protected String[] inputNames;
	protected String[] outputNames;
	protected Attributes attributes;

	public Node(Model model, NodeProto nodeProto, Tensor.Options tensorOptions) {
		super(nodeProto.getName(), nodeProto.getDocString());

		this.inputNames = nodeProto.getInputList().toArray(new String[nodeProto.getInputList().size()]);
		this.outputNames = nodeProto.getOutputList().toArray(new String[nodeProto.getOutputList().size()]);

		this.domain = nodeProto.getDomain();
		this.opType = nodeProto.getOpType();
		this.attributes = new Attributes(model, nodeProto.getAttributeList());
	}

	public String[] getInputNames() {
		return inputNames;
	}

	public String[] getOutputNames() {
		return outputNames;
	}

	public Attributes getAttrs() {
		return attributes;
	}

	public String getOpType() {
		return opType;
	}

	@Override
	public String toString() {
		return "Node [domain=" + domain + ", opType=" + opType + ", inputNames=" + Arrays.toString(inputNames)
				+ ", outputNames=" + Arrays.toString(outputNames) + ", attributes=" + attributes + "]";
	}

}