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
package org.onnx4j.opsets.domain.aiOnnx.v1.ops;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Shape Operator v1
 * 
 * <p>
 * Takes a tensor as input and outputs an 1D int64 tensor containing the shape
 * of the input tensor.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Shape-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape"> ONNX
 *      .Operators.md</a>
 */
public interface ShapeV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Shape";

	/**
	 * Input tensor can be of arbitrary type.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.allTypes());

	/**
	 * Constrain output to int64 tensor.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T1 = new TypeConstraint(DataType.INT64);

	/**
	 * Executes operator
	 * 
	 * @param x
	 *            Input tensor
	 * @return Output tensor
	 */
	// public T_TENSOR shape(T_TENSOR data);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	class ShapeInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		protected Field<T_TENSOR> dataField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

		public ShapeInputsV1(Node node, Inputs inputs) {
			super(node, inputs);
		}

		public T_TENSOR getData() {
			return dataField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ShapeOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public ShapeOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T1;
		}

	}

}