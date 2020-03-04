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

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.model.graph.node.attributes.StringAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * MaxPool Operator v1
 * 
 * <p>
 * MaxPool consumes an input tensor X and applies max pooling across the tensor
 * according to kernel sizes, stride sizes, and pad lengths. max pooling
 * consisting of computing the max on all values of a subset of the input tensor
 * according to the kernel size and downsampling the data into the output tensor
 * Y for further processing.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#MaxPool-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool">
 *      ONNX.Operators.md</a>
 */
public interface MaxPoolV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "MaxPool";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            Input data tensor from the previous operator; dimensions for
	 *            image case are (N x C x H x W), where N is the batch size, C
	 *            is the number of channels, and H and W are the height and the
	 *            width of the data. For non image case, the dimensions are in
	 *            the form of (N x C x D1 x D2 ... Dn), where N is the batch
	 *            size. Optionally, if dimension denotation is in effect, the
	 *            operation expects the input data tensor to arrive with the
	 *            dimension denotation of [DATA_BATCH, DATA_CHANNEL,
	 *            DATA_FEATURE, DATA_FEATURE ...].
	 * @param autoPad
	 *            auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or
	 *            VALID. Where default value is NOTSET, which means explicit
	 *            padding is used. SAME_UPPER or SAME_LOWER mean pad the input
	 *            so that the output spatial size match the input.In case of odd
	 *            number add the extra padding at the end for SAME_UPPER and at
	 *            the beginning for SAME_LOWER. VALID mean no padding.
	 * @param kernelShape
	 *            The size of the kernel along each axis.
	 * @param pads
	 *            Padding for the beginning and ending along each spatial axis,
	 *            it can take any value greater than or equal to 0. The value
	 *            represent the number of pixels added to the beginning and end
	 *            part of the corresponding axis. `pads` format should be as
	 *            follow [x1_begin, x2_begin...x1_end, x2_end,...], where
	 *            xi_begin the number of pixels added at the beginning of axis
	 *            `i` and xi_end, the number of pixels added at the end of axis
	 *            `i`. This attribute cannot be used simultaneously with
	 *            auto_pad attribute. If not present, the padding defaults to 0
	 *            along start and end of each spatial axis.
	 * @param strides
	 *            Stride along each spatial axis.
	 * @return Output data tensor from average or max pooling across the input
	 *         tensor. Dimensions will vary based on various kernel, stride, and
	 *         pad sizes. Floor value of the dimension is used
	 */
	/*public abstract T_TENSOR maxpool(T_TENSOR data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides);*/

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	/**
	 * Inputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class MaxPoolInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_AUTO_PAD = "auto_pad";

		public static final String ATTR_KERNEL_SHAPE = "kernel_shape";

		public static final String ATTR_PADS = "pads";

		public static final String ATTR_STRIDES = "strides";

		protected Field<T_TENSOR> dataField;
		protected Field<String> autoPadField;
		protected Field<List<Long>> kernelShapeField;
		protected Field<List<Long>> padsField;
		protected Field<List<Long>> stridesField;

		public MaxPoolInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			dataField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			//
			// string (default is NOTSET)
			//
			autoPadField = new AttributeField<String>(super.attrs, ATTR_AUTO_PAD, StringAttribute.class, "NOTSET",
					false);

			//
			// list of ints (required)
			//
			kernelShapeField = new AttributeField<List<Long>>(super.attrs, ATTR_KERNEL_SHAPE, IntsAttribute.class, null,
					true);

			//
			// list of ints
			//
			padsField = new AttributeField<List<Long>>(super.attrs, ATTR_PADS, IntsAttribute.class, null, false);

			//
			// list of ints
			//
			stridesField = new AttributeField<List<Long>>(super.attrs, ATTR_STRIDES, IntsAttribute.class, null, false);
		}

		public T_TENSOR getData() {
			return dataField.getData();
		}

		public String getAutoPad() {
			return autoPadField.getData();
		}

		public List<Long> getKernelShape() {
			return kernelShapeField.getData();
		}

		public List<Long> getStrides() {
			return stridesField.getData();
		}

		public List<Long> getPads() {
			return padsField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward) s
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class MaxPoolOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public MaxPoolOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}