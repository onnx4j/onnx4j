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
package org.onnx4j.opsets.aiOnnx.v1.ops;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Computes the indices of the max elements of the input tensor's element along
 * the provided axis. The resulted tensor has the same rank as the input if
 * keepdims equal 1. If keepdims equal 0, then the resulted tensor have the
 * reduced dimension pruned. The type of the output tensor is integer.
 * 
 * @author HarryLee
 * 
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 * @param <T_TENSOR>
 */
public interface ArgMaxV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "ArgMax";

	//
	// The axis in which to compute the arg indices.
	//
	public static final String ATTR_AXIS = "axis";

	//
	// Keep the reduced dimension or not, default 1 mean keep reduced dimension.
	//
	public static final String ATTR_KEEPDIMS = "keepdims";

	public abstract T_TENSOR argmax(T_TENSOR x0, int axis, int keepdims);
	
	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.numericTypes();
	}

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Attributes attrs = node.getAttrs();
		
		//
		// axis : int (default is 0)
		//
		int axis = attrs.getAttrValue(ATTR_AXIS, IntAttribute.class, 0L).intValue();

		//
		// keepdims : int (default is 1)
		//
		int keepdims = attrs.getAttrValue(ATTR_KEEPDIMS, IntAttribute.class, 1L).intValue();
		
		Input[] inputArray = inputs.get();
		return Outputs.wrap(node, this.argmax(inputArray[0].getTensor(), axis, keepdims));
	}

}