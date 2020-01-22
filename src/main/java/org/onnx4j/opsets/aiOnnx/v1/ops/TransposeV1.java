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
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.OperatorInputs;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

import com.google.common.collect.Lists;

/**
 * Transpose Operator v1
 * 
 * <p>
 * Transpose the input tensor similar to numpy.transpose. For example, when
 * perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
 * will be (2, 1, 3).
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Transpose-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose">
 *      ONNX.Operators.md</a>
 */
public interface TransposeV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Transpose";

	class TransposeInputsV1<T_TENSOR> extends OperatorInputs {

		//
		// list of ints
		//
		public static final String ATTR_PERM = "perm";

		protected T_TENSOR data;
		protected List<Long> perm;

		public TransposeInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			Attributes attrs = node.getAttrs();

			this.data = super.inputArray[0].getTensor();
			this.perm = attrs.getAttrValue(ATTR_PERM, IntsAttribute.class, Lists.newLinkedList());
		}

		public T_TENSOR getData() {
			return data;
		}

		public List<Long> getPerm() {
			return perm;
		}

	}

	/**
	 * Executes operator
	 * 
	 * @param x
	 *            Input tensor
	 * @return Output tensor
	 */
	public T_TENSOR transpose(T_TENSOR data, List<Long> perm);

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		TransposeInputsV1<T_TENSOR> operatorInputs = new TransposeInputsV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.transpose(operatorInputs.getData(), operatorInputs.getPerm()));
	}

}