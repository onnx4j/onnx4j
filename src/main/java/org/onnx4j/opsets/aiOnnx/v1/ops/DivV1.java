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

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Div-1
 * 
 * Performs element-wise binary division (with limited broadcast support).
 * 
 * If necessary the right-hand-side argument will be broadcasted to match the
 * shape of left-hand-side argument. When broadcasting is specified, the second
 * tensor can either be of element size 1 (including a scalar tensor and any
 * tensor with rank equal to or smaller than the first tensor), or having its
 * shape as a contiguous subset of the first tensor's shape. The starting of the
 * mutually equal shape is specified by the argument "axis", and if it is not
 * set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 * 
 * For example, the following tensor shapes are supported (with broadcast=1):
 * 
 * <pre>
 * shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
 * shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
 * shape(A) = (2, 3, 4, 5), shape(B) = (5,)
 * shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
 * shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
 * shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
 * </pre>
 * 
 * Attribute broadcast=1 needs to be passed to enable broadcasting.
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Div-1
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface DivV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Div";

	//
	// If set, defines the broadcast dimensions. See doc for details.
	//
	public static final String ATTR_AXIS = "axis";

	//
	// Pass 1 to enable broadcasting.
	//
	public static final String ATTR_BROADCAST = "broadcast";

	//
	// Legacy optimization attribute.
	//
	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

	/**
	 * Execute div operator
	 * 
	 * @param a
	 *            First operand, should share the type with the second operand.
	 * @param b
	 *            Second operand. With broadcasting can be of smaller size than
	 *            A. If broadcasting is disabled it should be of the same size.
	 * @return Result, has same dimensions and type as A
	 */
	public abstract T_TENSOR div(T_TENSOR a, T_TENSOR b, Long axis, Long broadcast, List<Long> consumedInputs);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
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
		Long axis = attrs.getAttrValue(ATTR_AXIS, IntAttribute.class, 0L);

		//
		// boardcast : int (default is 0)
		//
		Long broadcast = attrs.getAttrValue(ATTR_BROADCAST, IntAttribute.class, 0L);

		//
		// Legacy optimization attribute.
		//
		List<Long> consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node,
				this.div(inputArray[0].getTensor(), inputArray[1].getTensor(), axis, broadcast, consumedInputs));
	}

}