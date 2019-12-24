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
package org.onnx4j.opsets.aiOnnx.v6.ops;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.opsets.aiOnnx.v1.ops.MulV1;

/**
 * Mul Operator v6
 * 
 * <p>
 * Performs element-wise binary multiplication (with limited broadcast support).
 * 
 * If necessary the right-hand-side argument will be broadcasted to match the
 * shape of left-hand-side argument. When broadcasting is specified, the second
 * tensor can either be of element size 1 (including a scalar tensor and any
 * tensor with rank equal to or smaller than the first tensor), or having its
 * shape as a contiguous subset of the first tensor's shape. The starting of the
 * mutually equal shape is specified by the argument "axis", and if it is not
 * set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 6
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Mul-6">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul">ONNX.
 *      Operators.md</a>
 */
public interface MulV6<T_TENSOR> extends MulV1<T_TENSOR> {

	public static final String OP_TYPE = "Mul";

	public static final String ATTR_AXIS = "axis";

	public static final String ATTR_BROADCAST = "broadcast";

	/**
	 * Executes operator
	 * 
	 * @param a
	 *            First operand, should share the type with the second operand.
	 * @param b
	 *            Second operand. With broadcasting can be of smaller size than
	 *            A. If broadcasting is disabled it should be of the same size.
	 * @param axis
	 *            If set, defines the broadcast dimensions. See doc for details.
	 * @param broadcast
	 *            Pass 1 to enable broadcasting.
	 * @return Result, has same dimensions and type as A
	 */
	public abstract T_TENSOR mul(T_TENSOR a, T_TENSOR b, Long axis, Long broadcast);

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Attributes attrs = node.getAttrs();

		//
		// If set, defines the broadcast dimensions. See doc for details.
		//
		Long axis = attrs.getAttrValue(ATTR_AXIS, IntAttribute.class, null);

		//
		// Pass 1 to enable broadcasting (default is 0)
		//
		Long broadcast = attrs.getAttrValue(ATTR_AXIS, IntAttribute.class, 0L);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node, this.mul(inputArray[0].getTensor(), inputArray[1].getTensor(), axis, broadcast));
	}

}